"""悬索桥抗风易损性分析框架。

本模块给出抗风易损性算法的整体结构，并按中文注释列出主要步骤。
各函数仅提供接口和示例逻辑，供后续完善使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, uniform, lognorm, rv_frozen, kstest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor, as_completed
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import itertools
import logging


def define_random_variables(
    *,
    V_b_100: float = 25.8,
    z_ref: float = 1502.4,
    z_bridge: float = 1363.0,
    k_valley: float = 1.2,
    beta_coeff: float = 0.10,
    allow_override: bool = True,
    **kwargs: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, rv_frozen]]:
    """计算基本风速 ``U10`` 与迎风攻角 ``alpha`` 的统计量及分布。

    参数默认取自《公路桥梁抗风设计规范》（JTG D60-01-2004）。

    算法步骤：
    1. 海拔修正： ``V_b(z) = V_b_ref * (1 + 0.0007\,(z - z_ref))``；
       将 ``V_b_100`` 从 ``z_ref`` 修正至 ``z_bridge``。
    2. 乘以山谷系数 ``k_valley`` 得设计风速 ``V_design``。
    3. 假设极值服从 GumbelⅠ型分布，尺度 ``β = beta_coeff * V_design``，
       由 ``x_T = μ - β\ln[-\ln(1-1/T)]``(``T=100``) 反求位置参数 ``μ``。
       平均值 ``mean = μ + γβ``，标准差 ``std = πβ/√6``。
    4. 攻角 ``alpha`` 取均匀分布 ``U(0,3°)``。

    若 ``allow_override=True``，可通过 ``kwargs`` 覆盖上述参数。
    返回 ``mu``、``std`` 及 ``dists`` 三个字典。
    """

    if allow_override:
        V_b_100 = kwargs.get("V_b_100", V_b_100)
        z_ref = kwargs.get("z_ref", z_ref)
        z_bridge = kwargs.get("z_bridge", z_bridge)
        k_valley = kwargs.get("k_valley", k_valley)
        beta_coeff = kwargs.get("beta_coeff", beta_coeff)

    # 海拔修正与山谷系数
    V_b_z = V_b_100 * (1 + 0.0007 * (z_bridge - z_ref))
    V_design = V_b_z * k_valley

    # Gumbel 分布参数
    beta = beta_coeff * V_design
    mu_loc = V_design + beta * np.log(-np.log(1 - 1 / 100))
    mean_u10 = mu_loc + 0.5772156649 * beta
    std_u10 = np.pi * beta / np.sqrt(6.0)

    # 攻角分布
    mu_alpha = 1.5
    std_alpha = 3.0 / np.sqrt(12.0)

    mu = {"U10": mean_u10, "alpha": mu_alpha}
    std = {"U10": std_u10, "alpha": std_alpha}
    dists = {
        "U10": gumbel_r(loc=mu_loc, scale=beta),
        "alpha": uniform(loc=0.0, scale=3.0),
    }
    return mu, std, dists


def generate_ccd_samples(
    center: Dict[str, float],
    std: Dict[str, float],
    k: float = 1.0,
    alpha_star: float = np.sqrt(2.0),
    n_center: int = 3,
) -> pd.DataFrame:
    """按照中央复合设计生成样本点。

    自动支持任意维数，返回物理坐标 ``var`` 和编码坐标 ``x1``、``x2``、...。
    ``k`` 表示 CCD 半径与标准差的比例。
    """

    var_names = list(center.keys())
    n = len(var_names)
    delta = {v: k * std[v] for v in var_names}

    records: List[Dict[str, float]] = []

    # 中心点
    code_zero = {f"x{i+1}": 0.0 for i in range(n)}
    for _ in range(n_center):
        rec = {v: center[v] for v in var_names} | code_zero | {"type": "center"}
        records.append(rec)

    # 角点
    for signs in itertools.product([-1.0, 1.0], repeat=n):
        phys = {v: center[v] + s * delta[v] for v, s in zip(var_names, signs)}
        code = {f"x{i+1}": s for i, s in enumerate(signs)}
        records.append(phys | code | {"type": "corner"})

    # 轴点
    for i in range(n):
        for s in (-alpha_star, alpha_star):
            phys = {v: center[v] for v in var_names}
            phys[var_names[i]] += s * delta[var_names[i]]
            code = {f"x{j+1}": 0.0 for j in range(n)}
            code[f"x{i+1}"] = s
            records.append(phys | code | {"type": "axial"})

    df = pd.DataFrame(records)
    df.attrs["center"] = center
    df.attrs["delta"] = delta
    return df


def run_coupled_simulation(sample: Dict[str, float], *args, **kwargs) -> Dict[str, float]:
    """高保真流固耦合仿真占位函数。"""

    logging.debug("start FSI simulation %s", sample)
    # TODO: 接入 ANSYS 等软件实现真实仿真
    result = {"lambda": 1.0, "D": 0.0, "amax": 0.0}
    logging.debug("finish FSI simulation %s", result)
    return result


def run_simulations_async(samples: List[Dict[str, float]], max_workers: int = 8, timeout: float | None = None) -> List[Dict[str, float]]:
    """并行执行多组流固耦合仿真。

    通过 ``ProcessPoolExecutor`` 异步调度 ``run_coupled_simulation``，
    并收集结果字典列表。超时和异常将记录警告。
    """

    results: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_coupled_simulation, s): s for s in samples}
        for fut in as_completed(futures, timeout=timeout):
            sample = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # pragma: no cover - 占位异常处理
                logging.warning("仿真失败 %s: %s", sample, exc)
                res = {"lambda": np.nan, "D": np.nan, "amax": np.nan}
            results.append(res)
    return results



@dataclass
class QuadraticRSM:
    """二次响应面模型。"""

    coeff: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.poly = PolynomialFeatures(degree=2, include_bias=True)

        self.center: Dict[str, float] | None = None
        self.delta: Dict[str, float] | None = None
        self.var_names: List[str] | None = None
        self.model: Ridge | None = None

        self.samples: np.ndarray | None = None
        self.responses: np.ndarray | None = None

    def fit(
        self,
        samples: pd.DataFrame,
        responses: np.ndarray,
        center: Dict[str, float],
        delta: Dict[str, float],
        alpha: float = 0.0,
    ) -> "QuadraticRSM":

        """拟合多项式系数，可在此嵌入 NSGA3 优化逻辑。"""

        self.center = center
        self.delta = delta
        self.var_names = list(center.keys())
        X = self.poly.fit_transform(samples[[f"x{i+1}" for i in range(len(self.var_names))]].values)

        self.samples = X
        self.responses = responses

        self.model = Ridge(alpha=alpha, fit_intercept=False)
        self.model.fit(X, responses)
        self.coeff = self.model.coef_
        return self

    def _ensure_encoded(self, samples: pd.DataFrame) -> pd.DataFrame:
        if "x1" not in samples.columns:
            for i, v in enumerate(self.var_names):
                samples[f"x{i+1}"] = (samples[v] - self.center[v]) / self.delta[v]
        return samples

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        samples = self._ensure_encoded(samples.copy())
        X = self.poly.transform(samples[[f"x{i+1}" for i in range(len(self.var_names))]].values)
        return self.model.predict(X)

    def gradient(self, sample: pd.DataFrame) -> np.ndarray:
        """返回对物理变量的梯度。"""
        sample = self._ensure_encoded(sample.copy())
        x = sample[[f"x{i+1}" for i in range(len(self.var_names))]].values[0]

        grad_x = np.zeros_like(x)
        idx = 1
        for i in range(len(x)):
            grad_x[i] += self.coeff[idx]
            idx += 1
        for i in range(len(x)):
            grad_x[i] += 2 * self.coeff[idx] * x[i]
            idx += 1
            for j in range(i + 1, len(x)):
                grad_x[i] += self.coeff[idx] * x[j]
                grad_x[j] += self.coeff[idx] * x[i]
                idx += 1
        # 转换为物理变量梯度
        grad_physical = grad_x / np.array([self.delta[v] for v in self.var_names])
        return grad_physical


    def optimize(self) -> None:
        """利用 NSGA3 多目标优化响应面系数。"""

        if self.model is None:
            raise RuntimeError("请先调用 fit 生成初始模型")

        n_coef = self.poly.fit_transform([[0] * len(self.var_names)]).shape[1]

        class RSMProblem(ElementwiseProblem):
            def __init__(self, model: Ridge, X: np.ndarray, y: np.ndarray) -> None:
                super().__init__(n_var=n_coef, n_obj=2)
                self.model = model
                self.X = X
                self.y = y

            def _evaluate(self, x: np.ndarray, out: Dict[str, np.ndarray]) -> None:
                self.model.coef_ = x
                pred = self.model.predict(self.X)
                mse = ((pred - self.y) ** 2).mean()
                l2 = np.linalg.norm(x)
                out["F"] = np.array([mse, l2])

        problem = RSMProblem(self.model, self.samples, self.responses)


        ref_dirs = get_reference_directions("das-dennis", 2, n_points=60)
        algo = NSGA3(pop_size=60, ref_dirs=ref_dirs)

        res = minimize(problem, algo, ("n_gen", 50), verbose=False)

        coeff_best = res.X[np.argmin(res.F[:, 0])]
        self.model.coef_ = coeff_best
        self.coeff = coeff_best



class ReliabilitySolver:
    """可靠度计算器，可选择 FORM 或其他算法。"""

    def __init__(self, method: str = "HL-RF") -> None:
        self.method = method

    def solve(

        self, rsm: QuadraticRSM, dists: Dict[str, rv_frozen], max_iter: int = 20, tol: float = 1e-6
    ) -> Tuple[float, Dict[str, float], float]:
        """执行 FORM 计算，返回 β、设计点以及预测值。"""

        mu = np.array([dist.mean() for dist in dists.values()])
        sigma = np.array([dist.std() for dist in dists.values()])
        var_names = list(dists.keys())

        u = np.zeros_like(mu)

        for _ in range(max_iter):
            x = mu + sigma * u
            sample = pd.DataFrame([{v: val for v, val in zip(var_names, x)}])
            g = rsm.predict(sample)[0]
            grad_x = rsm.gradient(sample)
            grad_u = grad_x * sigma
            norm_grad = np.linalg.norm(grad_u)
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
        g_pred_design = rsm.predict(pd.DataFrame([design_point]))[0]
        beta = np.linalg.norm(u)
        return beta, design_point, g_pred_design



def update_sampling_center(
    center: Dict[str, float],
    g_true_center: float,
    design_point: Dict[str, float],
    g_pred_design: float,
) -> Dict[str, float]:
    """根据真实值与预测值在直线段上线性插值更新中心。"""


    if abs(g_true_center - g_pred_design) < 1e-6:
        return center

    ratio = g_true_center / (g_true_center - g_pred_design)
    return {k: center[k] + ratio * (design_point[k] - center[k]) for k in center}


def iterate_until_convergence(
    mu: Dict[str, float],
    std: Dict[str, float],

    dists: Dict[str, rv_frozen],
    max_iter: int = 10,
    tol: float = 1e-2,
) -> Tuple[QuadraticRSM, List[Dict[str, float]]]:
    """反复校正响应面直至设计点收敛。"""

    center = mu.copy()

    scale = 1.0

    design_history: List[Dict[str, float]] = []
    rsm = QuadraticRSM()

    for _ in range(max_iter):

        delta = {k: scale * std[k] for k in std}
        samples = generate_ccd_samples(center, std, k=scale)
        sim_results = run_simulations_async(samples.to_dict("records"))
        responses = np.array([r["lambda"] - 1 for r in sim_results])
        rsm.fit(samples, responses, center=center, delta=delta)
        solver = ReliabilitySolver()
        beta, design, g_pred_design = solver.solve(rsm, dists)

        design_history.append(design)
        if len(design_history) > 1:
            prev = np.array(list(design_history[-2].values()))
            curr = np.array(list(design.values()))
            if np.linalg.norm(curr - prev) < tol:
                break
        g_true_center = run_coupled_simulation(center)["lambda"] - 1
        center = update_sampling_center(center, g_true_center, design, g_pred_design)

        if abs(g_true_center) < 1e-3:
            break
        scale = max(scale * 0.8, 0.2)


    return rsm, design_history


def monte_carlo_capacity(
    rsm: QuadraticRSM, dists: Dict[str, rv_frozen], size: int = 1000
) -> np.ndarray:
    """基于响应面的大样本容量估计。"""


    alpha_samples = dists["alpha"].rvs(size=size)
    mu_u = dists["U10"].mean()
    std_u = dists["U10"].std()

    capacities = []
    for a in alpha_samples:
        def func(u: float) -> float:
            return rsm.predict(pd.DataFrame([{"U10": u, "alpha": a}]))[0]

        u_low = 1.0
        u_high = mu_u + 6 * std_u
        if func(u_low) <= 0:
            capacities.append(u_low)
            continue
        if func(u_high) >= 0:
            capacities.append(np.nan)
            continue
        try:
            cap = brentq(func, u_low, u_high)

        except ValueError:
            cap = np.nan
        capacities.append(cap)

    return np.array(capacities)


def fit_fragility_curve(capacity_samples: np.ndarray) -> Tuple[float, float, float, float]:
    """对容量样本进行对数正态分布拟合，返回标准差及 KS 值。"""

    capacity_samples = capacity_samples[~np.isnan(capacity_samples)]
    shape, loc, scale = lognorm.fit(capacity_samples, floc=0)
    theta = scale
    beta = shape
    ks_stat, _ = kstest(np.log(capacity_samples), "norm", args=(np.log(theta), beta))
    se_beta = beta / np.sqrt(2 * len(capacity_samples))
    return theta, beta, se_beta, ks_stat



def main() -> None:
    """程序主入口。"""
    mu, std, dists = define_random_variables()
    rsm, history = iterate_until_convergence(mu, std, dists)
    capacity = monte_carlo_capacity(rsm, dists)
    theta, beta, se_beta, ks = fit_fragility_curve(capacity)
    print(theta, beta, se_beta, ks)

if __name__ == "__main__":
    main()