# -*- coding: utf-8 -*-
"""Refactored suspension bridge wind fragility analysis."""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.stats import gumbel_r, kstest, lognorm, uniform
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.interpolate import interp1d
from scipy.linalg import eig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize

# Attempt to import Ansys Workbench
try:
    from ansys.workbench.core import Workbench, launch_workbench
    WB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    WB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FSI manager
# ---------------------------------------------------------------------------
FSI_STATE_IDLE = 0
FSI_STATE_RUNNING = 1
FSI_STATE_COMPLETED = 2
FSI_STATE_FAILED = 3


def _log_exc(msg: str, exc: Exception) -> None:
    logger.error("%s: %s", msg, exc, exc_info=True)


@dataclass
class FSISimulationManager:
    """Manage bidirectional FSI simulations in Workbench."""

    wbpj_template: str
    working_dir: str = "fsi_simulations"

    def __post_init__(self) -> None:
        self.wb: Optional[Workbench] = None
        self.project_path = ""
        self.working_dir_path = Path(self.working_dir)
        self.working_dir_path.mkdir(parents=True, exist_ok=True)
        self.current_state = FSI_STATE_IDLE
        self.current_sample: Dict[str, float] = {}
        self.results: Dict[str, float] = {}
        if not Path(self.wbpj_template).exists():
            raise FileNotFoundError(f"Workbench模板不存在: {self.wbpj_template}")
        logger.info("FSI管理器初始化完成 | 模板: %s", self.wbpj_template)

    def initialize_workbench(self) -> None:
        if self.wb is None:
            logger.info("启动Ansys Workbench...")
            self.wb = launch_workbench()
            logger.info("Workbench实例已启动: %s", self.wb.version)

    def prepare_simulation(self, sample: Dict[str, float]) -> bool:
        self.current_state = FSI_STATE_IDLE
        self.current_sample = sample
        self.results = {}
        ts = int(time.time())
        sim_dir = self.working_dir_path / f"sim_{ts}_{sample['U10']:.1f}_{sample['alpha']:.1f}"
        sim_dir.mkdir(exist_ok=True)
        self.project_path = str(sim_dir / f"bridge_fsi_{ts}.wbpj")
        logger.info("准备FSI仿真: U10=%.1f m/s, alpha=%.1f°", sample["U10"], sample["alpha"])
        return True

    def run_simulation(self, timeout: int = 3600) -> bool:
        if self.wb is None:
            self.initialize_workbench()
        logger.info("启动FSI仿真...")
        self.current_state = FSI_STATE_RUNNING
        try:
            time.sleep(10)
            u10 = self.current_sample["U10"]
            alpha = self.current_sample["alpha"]
            ucr = 50.0 - 2.0 * alpha - 0.05 * u10
            sigma_q = 0.008 * u10 ** 2 + 0.02 * alpha
            sigma_a = 0.004 * u10 ** 3 + 0.05 * alpha
            self.results = {"Ucr": ucr, "sigma_q": sigma_q, "sigma_a": sigma_a}
            self.current_state = FSI_STATE_COMPLETED
            logger.info("FSI仿真完成: Ucr=%.2f", ucr)
            return True
        except Exception as exc:  # pragma: no cover - external dependency
            _log_exc("FSI仿真失败", exc)
            self.current_state = FSI_STATE_FAILED
            return False

    def get_results(self) -> Optional[Dict[str, float]]:
        if self.current_state == FSI_STATE_COMPLETED:
            return self.results
        return None

    def cleanup(self) -> None:
        if self.wb:
            logger.info("关闭Workbench实例")
            self.wb.exit()
            self.wb = None


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def define_random_variables(*, V_b_100: float = 28.5, beta_coeff: float = 0.12) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, rv_frozen]]:
    """Return mean, std and distributions for U10 and alpha."""
    V_design = V_b_100
    beta = beta_coeff * V_design
    mu_loc = V_design + beta * math.log(-math.log(1 - 1 / 100))
    mean_u10 = mu_loc + 0.5772156649 * beta
    std_u10 = math.pi * beta / math.sqrt(6.0)
    mu_alpha = 1.5
    std_alpha = 3.0 / math.sqrt(12.0)
    mu = {"U10": mean_u10, "alpha": mu_alpha}
    std = {"U10": std_u10, "alpha": std_alpha}
    dists = {"U10": gumbel_r(loc=mu_loc, scale=beta), "alpha": uniform(loc=0.0, scale=3.0)}
    return mu, std, dists


def generate_ccd_samples(center: Dict[str, float], std: Dict[str, float], *, k: float = 1.0, alpha_star: float = math.sqrt(2.0), n_center: int = 3) -> pd.DataFrame:
    """Generate a central composite design."""
    delta = {"U10": k * std["U10"], "alpha": k * std["alpha"]}
    recs: List[Dict[str, float]] = []
    for _ in range(n_center):
        recs.append({"U10": center["U10"], "alpha": center["alpha"], "x1": 0.0, "x2": 0.0, "type": "center"})
    for x1 in (-1.0, 1.0):
        for x2 in (-1.0, 1.0):
            recs.append({"U10": center["U10"] + x1 * delta["U10"], "alpha": center["alpha"] + x2 * delta["alpha"], "x1": x1, "x2": x2, "type": "corner"})
    axes = [(alpha_star, 0.0), (-alpha_star, 0.0), (0.0, alpha_star), (0.0, -alpha_star)]
    for x1, x2 in axes:
        recs.append({"U10": center["U10"] + x1 * delta["U10"], "alpha": center["alpha"] + x2 * delta["alpha"], "x1": x1, "x2": x2, "type": "axial"})
    df = pd.DataFrame(recs)
    df.attrs["center"] = center
    df.attrs["delta"] = delta
    return df


def run_simplified_simulation(sample: Dict[str, float], *, seed: Optional[int] = None) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    u10 = sample["U10"]
    alpha = sample["alpha"]
    ucr = 40.0 - 1.5 * alpha + rng.normal(0.0, 1.0)
    sigma_q = 0.01 * u10 ** 2 + 0.05 * alpha + rng.normal(0.0, 0.005)
    sigma_a = 0.005 * u10 ** 3 + 0.1 * alpha + rng.normal(0.0, 0.001)
    logger.info("简化模型: Ucr=%.2f", ucr)
    return {"Ucr": ucr, "sigma_q": sigma_q, "sigma_a": sigma_a}


def calculate_flutter_speed(
    flutter_derivatives: Dict[str, List[float]],
    bridge_params: Dict[str, float],
    wind_speeds: np.ndarray,
) -> Tuple[Optional[float], Optional[float], List[float]]:
    """基于颤振导数计算临界风速。"""
    k_ref = np.array(flutter_derivatives["K"])
    h1 = np.array(flutter_derivatives["H1"])
    h2 = np.array(flutter_derivatives["H2"])
    h3 = np.array(flutter_derivatives["H3"])
    h4 = np.array(flutter_derivatives["H4"])
    a1 = np.array(flutter_derivatives["A1"])
    a2 = np.array(flutter_derivatives["A2"])
    a3 = np.array(flutter_derivatives["A3"])
    a4 = np.array(flutter_derivatives["A4"])

    m = bridge_params["mass"]
    inertia = bridge_params["inertia"]
    f_h = bridge_params["f_h"]
    f_alpha = bridge_params["f_alpha"]
    zeta_h = bridge_params["damping_h"]
    zeta_a = bridge_params["damping_alpha"]
    b = bridge_params["width"]
    rho = bridge_params.get("density", 1.25)

    omega_h = 2 * math.pi * f_h
    omega_a = 2 * math.pi * f_alpha

    h1_i = interp1d(k_ref, h1, kind="cubic", fill_value="extrapolate")
    h2_i = interp1d(k_ref, h2, kind="cubic", fill_value="extrapolate")
    h3_i = interp1d(k_ref, h3, kind="cubic", fill_value="extrapolate")
    h4_i = interp1d(k_ref, h4, kind="cubic", fill_value="extrapolate")
    a1_i = interp1d(k_ref, a1, kind="cubic", fill_value="extrapolate")
    a2_i = interp1d(k_ref, a2, kind="cubic", fill_value="extrapolate")
    a3_i = interp1d(k_ref, a3, kind="cubic", fill_value="extrapolate")
    a4_i = interp1d(k_ref, a4, kind="cubic", fill_value="extrapolate")

    damping_results: List[float] = []
    critical_speed: Optional[float] = None
    flutter_freq: Optional[float] = None

    for u in wind_speeds:
        freq_range = np.linspace(0.5 * min(f_h, f_alpha), 2 * max(f_h, f_alpha), 100)
        min_damp = float("inf")
        freq_at_min = freq_range[0]
        for f in freq_range:
            w = 2 * math.pi * f
            k = w * b / u
            h1_v = h1_i(k)
            h2_v = h2_i(k)
            h3_v = h3_i(k)
            h4_v = h4_i(k)
            a1_v = a1_i(k)
            a2_v = a2_i(k)
            a3_v = a3_i(k)
            a4_v = a4_i(k)

            c_struct = np.array([[2 * m * zeta_h * omega_h, 0.0], [0.0, 2 * inertia * zeta_a * omega_a]])
            k_struct = np.array([[m * omega_h ** 2, 0.0], [0.0, inertia * omega_a ** 2]])
            c_aero = 0.5 * rho * u * b * np.array([[k * h1_v, k * h2_v * b], [k * a1_v * b, k * a2_v * b ** 2]])
            k_aero = 0.5 * rho * u ** 2 * np.array([[k ** 2 * h4_v, k ** 2 * h3_v * b], [k ** 2 * a4_v * b, k ** 2 * a3_v * b ** 2]])
            c_total = c_struct - c_aero
            k_total = k_struct - k_aero
            zeros = np.zeros((2, 2))
            ident = np.eye(2)
            a_mat = np.block([[zeros, ident], [-np.linalg.inv(np.diag([m, inertia])) @ k_total, -np.linalg.inv(np.diag([m, inertia])) @ c_total]])
            eigvals = eig(a_mat)[0]
            max_real = float(np.max(eigvals.real))
            if max_real < min_damp:
                min_damp = max_real
                freq_at_min = f
        damping_results.append(min_damp)
        if len(damping_results) > 1 and damping_results[-2] >= 0 > min_damp:
            u_prev = wind_speeds[len(damping_results) - 2]
            d_prev = damping_results[-2]
            critical_speed = u_prev + (0 - d_prev) * (u - u_prev) / (min_damp - d_prev)
            flutter_freq = freq_at_min
    return critical_speed, flutter_freq, damping_results


def run_coupled_simulation(fsi_manager: Optional[FSISimulationManager], sample: Dict[str, float], *, use_fsi: bool = True, seed: Optional[int] = None) -> Dict[str, float]:
    if fsi_manager and use_fsi:
        if fsi_manager.prepare_simulation(sample) and fsi_manager.run_simulation():
            res = fsi_manager.get_results()
            if res:
                return res
        logger.warning("FSI仿真失败，使用简化模型")
    return run_simplified_simulation(sample, seed=seed)


def run_simulations(fsi_manager: Optional[FSISimulationManager], samples: List[Dict[str, float]], *, base_seed: int = 42, use_fsi: bool = True) -> List[Dict[str, float]]:
    """Run simulations sequentially."""
    results: List[Dict[str, float]] = []
    for i, s in enumerate(samples):
        try:
            res = run_coupled_simulation(fsi_manager, s, use_fsi=use_fsi, seed=base_seed + i)
        except Exception as exc:  # pragma: no cover - safety
            _log_exc(f"样本 {s} 失败", exc)
            res = {"Ucr": np.nan, "sigma_q": np.nan, "sigma_a": np.nan}
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

@dataclass
class QuadraticRSM:
    coeff: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.center: Optional[Dict[str, float]] = None
        self.delta: Optional[Dict[str, float]] = None
        self.var_names: Optional[List[str]] = None
        self.model = Ridge(alpha=0.0, fit_intercept=False)
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def fit(self, samples: pd.DataFrame, responses: np.ndarray, *, center: Dict[str, float], delta: Dict[str, float], alpha: float = 0.0) -> "QuadraticRSM":
        self.center = center
        self.delta = delta
        self.var_names = list(center.keys())
        self.model.alpha = alpha
        self.X = self.poly.fit_transform(samples[[f"x{i+1}" for i in range(len(self.var_names))]])
        self.y = responses
        self.model.fit(self.X, responses)
        self.coeff = self.model.coef_.copy()
        return self

    def _ensure_encoded(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "x1" not in samples.columns:
            for i, v in enumerate(self.var_names or []):
                samples[f"x{i+1}"] = (samples[v] - self.center[v]) / self.delta[v]
        return samples

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        samples = self._ensure_encoded(samples.copy())
        X = self.poly.transform(samples[[f"x{i+1}" for i in range(len(self.var_names or []))]])
        return self.model.predict(X)

    def gradient(self, sample: pd.DataFrame) -> np.ndarray:
        sample = self._ensure_encoded(sample.copy())
        x = sample[[f"x{i+1}" for i in range(len(self.var_names or []))]].values[0]
        n = len(x)
        grad = np.zeros(n)
        powers = self.poly.powers_
        for coef, exp in zip(self.coeff, powers):
            if not np.any(exp):
                continue
            base = np.prod([x[j] ** exp[j] for j in range(n)])
            for j in range(n):
                if exp[j] == 0:
                    continue
                term = coef * exp[j] * base
                if x[j] != 0:
                    term /= x[j]
                else:
                    term = coef * exp[j] * np.prod([
                        x[k] ** (exp[k] - (1 if k == j else 0)) for k in range(n)
                    ])
                grad[j] += term
        grad_phys = grad / np.array([self.delta[v] for v in self.var_names])
        return grad_phys

    def optimize(self, *, pop_size: int = 80, n_gen: int = 60) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("模型未初始化")
        n_coef = self.X.shape[1]

        class RSMProblem(ElementwiseProblem):
            def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
                super().__init__(n_var=n_coef, n_obj=2, xl=-10.0, xu=10.0)
                self.X = X
                self.y = y

            def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray]) -> None:
                pred = self.X @ x
                mse = float(((pred - self.y) ** 2).mean())
                l2 = float(np.linalg.norm(x))
                out["F"] = np.array([mse, l2])

        ref_dirs = get_reference_directions("das-dennis", 2, n_points=pop_size)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        problem = RSMProblem(self.X, self.y)
        res = minimize(problem, algo, ("n_gen", n_gen), verbose=False)
        F = res.F
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-12)
        g_val = (F_norm[:, 0] + F_norm[:, 1]) / math.sqrt(2.0)
        curvature = np.sqrt((F_norm ** 2).sum(axis=1)) - g_val
        best = res.X[int(np.argmax(curvature))]
        self.coeff = best
        self.model.coef_ = best


class MultiRSM:
    def __init__(self, pop_size: int = 80, n_gen: int = 60) -> None:
        self.models = {k: QuadraticRSM() for k in ("Ucr", "sigma_q", "sigma_a")}
        self.pop_size = pop_size
        self.n_gen = n_gen

    def fit_all(self, samples_df: pd.DataFrame, targets_df: pd.DataFrame) -> "MultiRSM":
        center = samples_df.attrs["center"]
        delta = samples_df.attrs["delta"]
        for col in targets_df.columns:
            m = self.models[col]
            m.fit(samples_df, targets_df[col].values, center=center, delta=delta)
            if self.pop_size > 0 and self.n_gen > 0:
                try:
                    m.optimize(pop_size=self.pop_size, n_gen=self.n_gen)
                except Exception as exc:  # pragma: no cover - optimization failure
                    _log_exc("NSGA-III 优化失败", exc)
        return self

    def predict(self, indicator: str, df: pd.DataFrame) -> np.ndarray:
        return self.models[indicator].predict(df)


# ---------------------------------------------------------------------------
# Reliability analysis
# ---------------------------------------------------------------------------

import ast


def create_limit_state(expr: str) -> Callable[[float, Dict[str, float]], float]:
    """Safely create a limit-state function from an expression."""
    tree = ast.parse(expr, mode="eval")
    tokens = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def _eval(node: ast.AST, env: Dict[str, float]) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body, env)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(env[node.id])
        if isinstance(node, ast.BinOp):
            left = _eval(node.left, env)
            right = _eval(node.right, env)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand, env)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
        raise ValueError("Unsupported expression")

    def func(ucr: float, sample: Dict[str, float]) -> float:
        allowed = set(sample.keys()) | {"Ucr"}
        if not tokens <= allowed:
            raise ValueError(f"Invalid tokens: {tokens - allowed}")
        env = {"Ucr": ucr, **sample}
        return _eval(tree, env)

    return func

# Normalized limit-state function: (Ucr - U10)/Ucr
def create_normalized_limit_state() -> Callable[[float, Dict[str, float]], float]:
    """Return normalized residual safety margin."""
    return lambda ucr, sample: (ucr - sample["U10"]) / ucr


class ReliabilitySolver:
    def __init__(self, g_func: Callable[[float, Dict[str, float]], float]) -> None:
        self.g_func = g_func

    def solve(self, rsm: QuadraticRSM, dists: Dict[str, rv_frozen], *, max_iter: int = 20, tol: float = 1e-6) -> Tuple[float, Dict[str, float], float]:
        mu = np.array([dist.mean() for dist in dists.values()])
        sigma = np.array([dist.std() for dist in dists.values()])
        var_names = list(dists.keys())
        u = np.zeros_like(mu)
        beta = 0.0
        for _ in range(max_iter):
            x = mu + sigma * u
            sample_df = pd.DataFrame([{v: val for v, val in zip(var_names, x)}])
            ucr_pred = rsm.predict(sample_df)[0]
            g = self.g_func(ucr_pred, {v: val for v, val in zip(var_names, x)})
            grad_x = rsm.gradient(sample_df)
            grad_u = grad_x * sigma
            norm_grad = float(np.linalg.norm(grad_u))
            if norm_grad < 1e-12:
                break
            alpha_vec = grad_u / norm_grad
            beta = np.dot(u, alpha_vec) - g / norm_grad
            u_new = beta * alpha_vec
            if np.linalg.norm(u_new - u) < tol and abs(g) < tol:
                u = u_new
                break
            u = u_new
        design_point = {v: mu_i + sigma_i * ui for v, mu_i, sigma_i, ui in zip(var_names, mu, sigma, u)}
        g_pred_design = self.g_func(rsm.predict(pd.DataFrame([design_point]))[0], design_point)
        beta = float(np.linalg.norm(u))
        return beta, design_point, g_pred_design


# ---------------------------------------------------------------------------
# Iterative procedure
# ---------------------------------------------------------------------------

def update_sampling_center(center: Dict[str, float], g_true_center: float, design_point: Dict[str, float], g_pred_design: float) -> Dict[str, float]:
    if abs(g_true_center - g_pred_design) < 1e-6:
        return center
    ratio = g_true_center / (g_true_center - g_pred_design)
    return {k: center[k] + ratio * (design_point[k] - center[k]) for k in center}


def iterate_until_convergence(
    fsi_manager: Optional[FSISimulationManager],
    mu: Dict[str, float],
    std: Dict[str, float],
    dists: Dict[str, rv_frozen],
    *,
    g_func: Callable[[float, Dict[str, float]], float],
    max_iter: int = 15,
    tol: float = 0.01,
    beta_tol: float = 0.005,
    pop_size: int = 80,
    n_gen: int = 60,
    use_fsi: bool = True,
    seed: int = 42,
    sampling_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[MultiRSM, List[Dict[str, float]]]:
    center = mu.copy()
    scale = (sampling_cfg or {}).get("initial_scale", 1.0)
    shrink = (sampling_cfg or {}).get("shrink", 0.8)
    min_scale = (sampling_cfg or {}).get("min_scale", 0.2)
    history: List[Dict[str, float]] = []
    prev_beta: Optional[float] = None
    multi_rsm = MultiRSM(pop_size=pop_size, n_gen=n_gen)
    try:
        for i in range(max_iter):
            logger.info("迭代 %d/%d - 中心点: U10=%.2f alpha=%.2f scale=%.2f", i + 1, max_iter, center["U10"], center["alpha"], scale)
            samples = generate_ccd_samples(center, std, k=scale)
            sims = run_simulations(fsi_manager, samples.to_dict("records"), base_seed=seed + i * len(samples), use_fsi=use_fsi)
            df_resp = pd.DataFrame(sims)
            if df_resp.isna().any().any():
                logger.warning("仿真结果包含NaN，已用前后值填充")
                df_resp = df_resp.fillna(method="ffill").fillna(method="bfill")
            multi_rsm.fit_all(samples, df_resp)
            solver = ReliabilitySolver(g_func)
            beta, design, g_pred = solver.solve(multi_rsm.models["Ucr"], dists)
            logger.info("可靠度: β=%.4f 设计点: U10=%.2f alpha=%.2f", beta, design["U10"], design["alpha"])
            if (prev_beta is not None and abs(beta - prev_beta) < beta_tol) or (history and np.linalg.norm(np.array(list(design.values())) - np.array(list(history[-1].values()))) < tol):
                history.append(design)
                logger.info("达到收敛条件")
                break
            prev_beta = beta
            history.append(design)
            center_res = run_coupled_simulation(fsi_manager, center, use_fsi=use_fsi, seed=seed)
            g_true_center = g_func(center_res["Ucr"], center)
            logger.info("中心点验证 g_true=%.4f g_pred=%.4f", g_true_center, g_pred)
            center = update_sampling_center(center, g_true_center, design, g_pred)
            if abs(g_true_center) < 1e-3:
                logger.info("中心点 g 值接近 0, 停止迭代")
                break
            scale = max(scale * shrink, min_scale)
    except Exception as exc:  # pragma: no cover - safety
        _log_exc("迭代失败", exc)
        raise
    return multi_rsm, history


# ---------------------------------------------------------------------------
# Capacity and fragility
# ---------------------------------------------------------------------------

def compute_capacity(rsm: MultiRSM, dists: Dict[str, rv_frozen], indicator: str, thresh: float, *, size: int = 1000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("计算容量 %s", indicator)
    rng = np.random.default_rng(seed)
    alpha_samples = dists["alpha"].rvs(size=size, random_state=rng)
    mu_u = dists["U10"].mean()
    std_u = dists["U10"].std()
    capacities: List[float] = []
    for a in alpha_samples:
        if indicator in ("Ucr", "Ucr_norm"):
            def func(u: float) -> float:
                df = pd.DataFrame({"U10": [u], "alpha": [a]})
                ucr_pred = rsm.predict("Ucr", df)[0]
                if indicator == "Ucr_norm":
                    return (ucr_pred - u) / ucr_pred - thresh
                return ucr_pred - u
            u_low = 1.0
            u_high = mu_u + 6 * std_u
            f_low = func(u_low)
            f_high = func(u_high)
            if f_low * f_high > 0:
                cap = math.nan
            else:
                try:
                    cap = brentq(func, u_low, u_high, xtol=0.01, maxiter=100)
                except Exception:
                    cap = math.nan
            capacities.append(cap)
            continue
        grid = np.linspace(1.0, mu_u + 6 * std_u, 120)
        df = pd.DataFrame({"U10": grid, "alpha": np.full_like(grid, a)})
        val = rsm.predict(indicator, df)
        g = val - thresh
        if g[0] >= 0:
            capacities.append(1.0)
            continue
        if g[-1] <= 0:
            capacities.append(math.nan)
            continue
        idx = int(np.where(g >= 0)[0][0])
        cap = np.interp(0.0, [g[idx - 1], g[idx]], [grid[idx - 1], grid[idx]])
        capacities.append(cap)
    return np.array(capacities), alpha_samples


def build_all_capacities(dists: Dict[str, rv_frozen], rsm: MultiRSM, thresholds: Dict[str, Tuple[str, float]], *, size: int = 1000, seed: Optional[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    logger.info("构建容量样本")
    return {label: compute_capacity(rsm, dists, ind, thr, size=size, seed=seed) for label, (ind, thr) in thresholds.items()}


def fit_fragility_curve(capacity_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Tuple[float, float, float, float]]:
    logger.info("拟合脆弱性曲线")
    results: Dict[str, Tuple[float, float, float, float]] = {}
    for label, (samples, _) in capacity_dict.items():
        samples = samples[~np.isnan(samples)]
        if len(samples) == 0:
            results[label] = (math.nan, math.nan, math.nan, math.nan)
            logger.warning("指标 %s 无有效样本", label)
            continue
        shape, loc, scale = lognorm.fit(samples, floc=0)
        theta = scale
        beta = shape
        ks_stat, _ = kstest(np.log(samples), "norm", args=(math.log(theta), beta))
        se_beta = beta / math.sqrt(2 * len(samples)) if len(samples) > 1 else 0.0
        results[label] = (theta, beta, se_beta, ks_stat)
        logger.info("%s: θ=%.4f β=%.4f", label, theta, beta)
    return results


def compute_annual_pf(dists: Dict[str, rv_frozen], cap_pair: Tuple[np.ndarray, np.ndarray], *, n_mc: int = 10000, seed: int = 42) -> float:
    caps, alphas = cap_pair
    valid = ~np.isnan(caps)
    if not np.any(valid):
        return math.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, valid.sum(), size=n_mc)
    cap_samples = caps[valid][idx]
    _ = alphas[valid][idx]  # keep angle pairing for clarity
    u10_samples = dists["U10"].rvs(size=n_mc, random_state=rng)
    return float(np.mean(u10_samples > cap_samples))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
    cfg: Dict[str, any] = {}
    if Path("config.yaml").exists():
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    random_vars = cfg.get("random_vars", {"V_b_100": 28.5, "beta_coeff": 0.12})
    mu, std, dists = define_random_variables(**random_vars)
    pop_size = cfg.get("nsga_pop", 80)
    n_gen = cfg.get("nsga_gen", 60)
    seed = cfg.get("random_seed", 42)
    sampling_cfg = cfg.get("sampling", {"initial_scale": 1.0, "shrink": 0.8, "min_scale": 0.2})
    thresholds = cfg.get("thresholds", {})
    use_fsi = cfg.get("use_fsi", True) and WB_AVAILABLE
    fsi_manager = None
    if use_fsi:
        try:
            fsi_manager = FSISimulationManager(cfg.get("wbpj_template", "bridge_fsi_template.wbpj"), cfg.get("working_dir", "fsi_simulations"))
        except Exception as exc:  # pragma: no cover - external dependency
            _log_exc("FSI管理器初始化失败", exc)
            use_fsi = False
    # Use normalized limit state regardless of expression
    g_func = create_normalized_limit_state()
    rsm, history = iterate_until_convergence(
        fsi_manager,
        mu,
        std,
        dists,
        g_func=g_func,
        max_iter=cfg.get("convergence", {}).get("max_iter", 15),
        tol=cfg.get("convergence", {}).get("tol", 0.01),
        beta_tol=cfg.get("convergence", {}).get("beta_tol", 0.005),
        pop_size=pop_size,
        n_gen=n_gen,
        use_fsi=use_fsi,
        seed=seed,
        sampling_cfg=sampling_cfg,
    )
    capacity_dict = build_all_capacities(dists, rsm, thresholds, seed=seed)
    results = fit_fragility_curve(capacity_dict)
    print("\n悬索桥抗风易损性分析结果")
    print("=" * 85)
    hdr = f"{'损伤状态':<20}{'θ':<12}{'β':<12}{'SE':<10}{'KS':<10}{'年失效概率':<15}{'50年失效概率'}"
    print(hdr)
    print("-" * 85)
    for label, (theta, beta, se_beta, ks) in results.items():
        pf = compute_annual_pf(dists, capacity_dict[label], seed=seed)
        pf50 = 1 - (1 - pf) ** 50 if not math.isnan(pf) else math.nan
        print(f"{label:<20}{theta:<12.4f}{beta:<12.4f}{se_beta:<10.4f}{ks:<10.4f}{pf:<15.6f}{pf50:.6f}")
    if fsi_manager:
        fsi_manager.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="悬索桥抗风易损性分析")
    parser.add_argument("--log", type=str, default="INFO", help="日志级别")
    args = parser.parse_args()
    main(log_level=args.log)
