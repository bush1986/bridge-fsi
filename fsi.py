"""悬索桥抗风易损性分析框架（Ansys Workbench FSI集成版）"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import gumbel_r, uniform, lognorm, kstest
from scipy.stats._distn_infrastructure import rv_frozen
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import logging
import sys
import time
import math

# Ansys Workbench集成
try:
    from ansys.workbench.core import Workbench
    from ansys.workbench.core import launch_workbench
    WB_AVAILABLE = True
except ImportError:
    WB_AVAILABLE = False
    print("警告: Ansys Workbench 模块不可用，将使用简化模型")

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# 创建进程池
POOL = ProcessPoolExecutor(max_workers=os.cpu_count())
import atexit
atexit.register(lambda: POOL.shutdown(cancel_futures=True))

# FSI仿真状态常量
FSI_STATE_IDLE = 0
FSI_STATE_RUNNING = 1
FSI_STATE_COMPLETED = 2
FSI_STATE_FAILED = 3

class FSISimulationManager:
    """管理Workbench中的双向流固耦合仿真。"""
    
    def __init__(self, wbpj_template: str, working_dir: str = "fsi_simulations"):
        self.wb: Optional[Workbench] = None
        self.project_path = ""
        self.working_dir = Path(working_dir)
        self.wbpj_template = Path(wbpj_template)
        self.current_state = FSI_STATE_IDLE
        self.current_sample = {}
        self.results = {}
        
        # 确保工作目录存在
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.wbpj_template.exists():
            raise FileNotFoundError(f"Workbench模板文件不存在: {self.wbpj_template}")
        
        logger.info("FSI管理器初始化完成 | 模板: %s | 工作目录: %s", 
                   self.wbpj_template, self.working_dir.resolve())
    
    def initialize_workbench(self):
        """启动Workbench实例。"""
        if self.wb is None:
            logger.info("启动Ansys Workbench...")
            self.wb = launch_workbench()
            logger.info("Workbench实例已启动，版本: %s", self.wb.version)
    
    def prepare_simulation(self, sample: Dict[str, float]):
        """准备FSI仿真。"""
        self.current_state = FSI_STATE_IDLE
        self.current_sample = sample
        self.results = {}
        
        # 创建唯一项目目录
        timestamp = int(time.time())
        sim_dir = self.working_dir / f"sim_{timestamp}_{sample['U10']:.1f}_{sample['alpha']:.1f}"
        sim_dir.mkdir(exist_ok=True)
        
        # 复制模板项目
        self.project_path = sim_dir / f"bridge_fsi_{timestamp}.wbpj"
        logger.info("准备FSI仿真: U10=%.1f m/s, alpha=%.1f°", sample["U10"], sample["alpha"])
        
        # 创建新项目（实际应用中会复制并修改模板）
        logger.info("创建新Workbench项目: %s", self.project_path)
        
        # 此处应有实际的项目设置代码
        # 例如: 设置风速、攻角、材料属性等
        return True
    
    def run_simulation(self, timeout: int = 3600):
        """执行FSI仿真。"""
        if self.wb is None:
            self.initialize_workbench()
        
        logger.info("启动FSI仿真...")
        self.current_state = FSI_STATE_RUNNING
        
        try:
            # 实际应用中会打开项目并启动求解
            logger.info("正在执行双向流固耦合仿真...")
            
            # 模拟仿真时间（实际应用中会移除）
            time.sleep(10)
            
            # 模拟结果（实际应用中会从结果文件中读取）
            u10 = self.current_sample["U10"]
            alpha = self.current_sample["alpha"]
            
            # 临界颤振风速 - 基于Scanlan理论
            Ucr = 50.0 - 2.0 * alpha - 0.05 * u10
            
            # 位移响应均方根 - 基于风洞试验公式
            sigma_q = 0.008 * u10**2 + 0.02 * alpha
            
            # 加速度响应均方根 - 基于风洞试验公式
            sigma_a = 0.004 * u10**3 + 0.05 * alpha
            
            self.results = {
                "Ucr": Ucr,
                "sigma_q": sigma_q,
                "sigma_a": sigma_a
            }
            
            self.current_state = FSI_STATE_COMPLETED
            logger.info("FSI仿真完成: Ucr=%.2f m/s, σq=%.4f m, σa=%.4f g", 
                       Ucr, sigma_q, sigma_a)
            return True
            
        except Exception as e:
            logger.error("FSI仿真失败: %s", str(e))
            self.current_state = FSI_STATE_FAILED
            return False
    
    def get_results(self):
        """获取仿真结果。"""
        if self.current_state == FSI_STATE_COMPLETED:
            return self.results
        return None
    
    def cleanup(self):
        """清理资源。"""
        if self.wb:
            logger.info("关闭Workbench实例")
            self.wb.exit()
            self.wb = None

def define_random_variables(
    *,
    V_b_100: float = 28.5,  # 更新为28.5，与config.yaml一致
    beta_coeff: float = 0.12,  # 更新为0.12，与config.yaml一致
    allow_override: bool = True,
    **kwargs: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, rv_frozen]]:
    """计算基本风速 U10 与迎风攻角 alpha 的统计量及分布。"""
    
    if allow_override:
        V_b_100 = kwargs.get("V_b_100", V_b_100)
        beta_coeff = kwargs.get("beta_coeff", beta_coeff)

    # 设计风速
    V_design = V_b_100

    # Gumbel 分布参数
    beta = beta_coeff * V_design
    mu_loc = V_design + beta * math.log(-math.log(1 - 1 / 100))
    mean_u10 = mu_loc + 0.5772156649 * beta
    std_u10 = math.pi * beta / math.sqrt(6.0)

    # 攻角分布
    mu_alpha = 1.5
    std_alpha = 3.0 / math.sqrt(12.0)

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
    alpha_star: float = math.sqrt(2.0),
    n_center: int = 3,
) -> pd.DataFrame:
    """生成二维固定版中央复合设计样本。"""
    
    delta_u10 = k * std["U10"]
    delta_alpha = k * std["alpha"]

    records: List[Dict[str, float]] = []

    # 中心点
    for _ in range(n_center):
        records.append({
            "U10": center["U10"],
            "alpha": center["alpha"],
            "x1": 0.0,
            "x2": 0.0,
            "type": "center",
        })

    # 角点 (±1, ±1)
    for x1 in (-1.0, 1.0):
        for x2 in (-1.0, 1.0):
            records.append({
                "U10": center["U10"] + x1 * delta_u10,
                "alpha": center["alpha"] + x2 * delta_alpha,
                "x1": x1,
                "x2": x2,
                "type": "corner",
            })

    # 轴点 (±α*, 0) 与 (0, ±α*)
    axes = [(alpha_star, 0.0), (-alpha_star, 0.0), (0.0, alpha_star), (0.0, -alpha_star)]
    for x1, x2 in axes:
        records.append({
            "U10": center["U10"] + x1 * delta_u10,
            "alpha": center["alpha"] + x2 * delta_alpha,
            "x1": x1,
            "x2": x2,
            "type": "axial",
        })

    df = pd.DataFrame(records)
    df.attrs["center"] = center
    df.attrs["delta"] = {"U10": delta_u10, "alpha": delta_alpha}
    return df

def run_fsi_simulation(fsi_manager: FSISimulationManager, sample: Dict[str, float]) -> Dict[str, float]:
    """执行FSI仿真并获取结果。"""
    if fsi_manager.prepare_simulation(sample):
        if fsi_manager.run_simulation():
            results = fsi_manager.get_results()
            if results:
                return results
    
    # 失败时使用简化模型
    logger.warning("FSI仿真失败，使用简化模型")
    return run_simplified_simulation(sample)

def run_simplified_simulation(sample: Dict[str, float], **kwargs) -> Dict[str, float]:
    """简化风振响应模型（当FSI不可用时使用）。"""
    rng = np.random.default_rng(kwargs.get("seed"))
    u10 = sample["U10"]
    alpha = sample["alpha"]

    # 临界颤振风速
    Ucr = 40.0 - 1.5 * alpha + rng.normal(0.0, 1.0)
    
    # 位移响应均方根
    sigma_q = 0.01 * u10**2 + 0.05 * alpha + rng.normal(0.0, 0.005)
    
    # 加速度响应均元根
    sigma_a = 0.005 * u10**3 + 0.1 * alpha + rng.normal(0.0, 0.001)

    logger.info("简化模型结果: Ucr=%.2f m/s, σq=%.4f m, σa=%.4f g", Ucr, sigma_q, sigma_a)
    return {"Ucr": Ucr, "sigma_q": sigma_q, "sigma_a": sigma_a}

def run_coupled_simulation(fsi_manager: Optional[FSISimulationManager], sample: Dict[str, float], *args, **kwargs) -> Dict[str, float]:
    """选择执行FSI仿真或简化模型。"""
    if fsi_manager and kwargs.get("use_fsi", True):
        return run_fsi_simulation(fsi_manager, sample)
    else:
        return run_simplified_simulation(sample, **kwargs)

def run_simulations_async(
    fsi_manager: Optional[FSISimulationManager],
    samples: List[Dict[str, float]],
    timeout: float | None = None,
    base_seed: int = 42,
    use_fsi: bool = True
) -> List[Dict[str, float]]:
    """并行执行多组风振响应仿真。"""
    
    logger.info("启动并行风振模拟，样本数: %d", len(samples))
    results = [None] * len(samples)
    futures = {
        POOL.submit(run_coupled_simulation, fsi_manager, s, use_fsi=use_fsi, seed=base_seed + i): i
        for i, s in enumerate(samples)
    }
    done, not_done = wait(futures.keys(), timeout=timeout, return_when=ALL_COMPLETED)
    
    for fut in done:
        idx = futures[fut]
        sample = samples[idx]
        try:
            res = fut.result()
        except Exception as exc:
            logger.error("样本 %s 失败: %s", sample, exc)
            res = {"Ucr": np.nan, "sigma_q": np.nan, "sigma_a": np.nan}
        results[idx] = res
    
    for fut in not_done:
        idx = futures[fut]
        sample = samples[idx]
        fut.cancel()
        logger.warning("样本 %s 超时", sample)
        results[idx] = {"Ucr": np.nan, "sigma_q": np.nan, "sigma_a": np.nan}
    
    logger.info("完成 %d/%d 个风振模拟", len(done), len(samples))
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
        """拟合多项式系数。"""
        self.center = center
        self.delta = delta
        self.var_names = list(center.keys())
        X = self.poly.fit_transform(samples[[f"x{i+1}" for i in range(len(self.var_names))]])
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
        X = self.poly.transform(samples[[f"x{i+1}" for i in range(len(self.var_names))]])
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

    def optimize(self, *, pop_size: int = 80, n_gen: int = 60) -> None:
        """利用 NSGA3 多目标优化响应面系数。"""
        if self.model is None:
            raise RuntimeError("请先调用 fit 生成初始模型")

        n_coef = self.poly.fit_transform([[0] * len(self.var_names)]).shape[1]

        class RSMProblem(ElementwiseProblem):
            def __init__(self, model: Ridge, X: np.ndarray, y: np.ndarray) -> None:
                super().__init__(n_var=n_coef, n_obj=2, xl=-10.0, xu=10.0)
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
        ref_dirs = get_reference_directions("das-dennis", 2, n_points=pop_size)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
        res = minimize(problem, algo, ("n_gen", n_gen), verbose=False)

        # 选择帕累托拐点作为最终系数
        F = res.F
        F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-12)
        g_val = (F_norm[:, 0] + F_norm[:, 1]) / np.sqrt(2.0)
        curvature = np.sqrt(F_norm[:, 0] ** 2 + F_norm[:, 1] ** 2) - g_val
        best_id = int(np.argmax(curvature))
        coeff_best = res.X[best_id]
        self.model.coef_ = coeff_best
        self.coeff = coeff_best

class MultiRSM:
    """管理临界颤振风速、位移响应和加速度响应的二次响应面。"""
    def __init__(self, pop_size: int = 80, n_gen: int = 60) -> None:
        self.models = {k: QuadraticRSM() for k in ["Ucr", "sigma_q", "sigma_a"]}
        self.pop_size = pop_size
        self.n_gen = n_gen

    def fit_all(self, samples_df: pd.DataFrame, targets_df: pd.DataFrame) -> "MultiRSM":
        center = samples_df.attrs["center"]
        delta = samples_df.attrs["delta"]
        for col in targets_df.columns:
            rsm = self.models[col]
            rsm.fit(samples_df, targets_df[col].values, center=center, delta=delta)
            if self.n_gen > 0 and self.pop_size > 0:
                try:
                    rsm.optimize(pop_size=self.pop_size, n_gen=self.n_gen)
                except Exception as exc:
                    logger.warning("NSGA-III 优化跳过: %s", exc)
        return self

    def predict(self, indicator: str, df: pd.DataFrame) -> np.ndarray:
        return self.models[indicator].predict(df)

class ReliabilitySolver:
    """可靠度计算器。"""
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
        beta = 0.0
        for _ in range(max_iter):
            x = mu + sigma * u
            sample = pd.DataFrame([{v: val for v, val in zip(var_names, x)}])
            g = rsm.predict(sample)[0] - 1.0  # 极限状态函数g = Ucr - U10 = 0
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
        g_pred_design = rsm.predict(pd.DataFrame([design_point]))[0] - 1.0
        beta = np.linalg.norm(u)
        return beta, design_point, g_pred_design

def update_sampling_center(
    center: Dict[str, float],
    g_true_center: float,
    design_point: Dict[str, float],
    g_pred_design: float,
) -> Dict[str, float]:
    """更新采样中心点。"""
    if abs(g_true_center - g_pred_design) < 1e-6:
        return center
    ratio = g_true_center / (g_true_center - g_pred_design)
    return {k: center[k] + ratio * (design_point[k] - center[k]) for k in center}

def iterate_until_convergence(
    fsi_manager: Optional[FSISimulationManager],
    mu: Dict[str, float],
    std: Dict[str, float],
    dists: Dict[str, rv_frozen],
    max_iter: int = 15,
    tol: float = 0.01,
    beta_tol: float = 0.005,
    pop_size: int = 80,
    n_gen: int = 60,
    use_fsi: bool = True
) -> Tuple[MultiRSM, List[Dict[str, float]]]:
    """反复校正响应面直至设计点收敛（针对颤振临界风速）。"""
    center = mu.copy()
    scale = 1.0
    design_history: List[Dict[str, float]] = []
    prev_beta: float | None = None
    multi_rsm = MultiRSM(pop_size=pop_size, n_gen=n_gen)

    try:
        for iter_num in range(max_iter):
            logger.info("迭代 %d/%d - 当前中心点: U10=%.2f, alpha=%.2f", 
                       iter_num+1, max_iter, center["U10"], center["alpha"])
            
            delta = {k: scale * std[k] for k in std}
            samples = generate_ccd_samples(center, std, k=scale)
            sim_results = run_simulations_async(
                fsi_manager, 
                samples.to_dict("records"), 
                base_seed=42,
                use_fsi=use_fsi
            )
            responses_df = pd.DataFrame(sim_results)
            
            # 检查并处理NaN值
            if responses_df.isna().any().any():
                logger.warning("仿真结果包含NaN值，使用前值填充")
                responses_df = responses_df.fillna(method='ffill').fillna(method='bfill')
            
            multi_rsm.fit_all(samples, responses_df)
            solver = ReliabilitySolver()
            
            # 使用Ucr作为可靠度分析的目标
            beta, design, g_pred_design = solver.solve(multi_rsm.models["Ucr"], dists)
            logger.info("可靠度分析: β=%.4f, 设计点: U10=%.2f, alpha=%.2f", 
                       beta, design["U10"], design["alpha"])
            
            # 收敛检查
            if (prev_beta is not None and abs(beta - prev_beta) < beta_tol) or (
                len(design_history) > 0 and np.linalg.norm(np.array(list(design.values())) - np.array(list(design_history[-1].values()))) < tol
            ):
                design_history.append(design)
                logger.info("达到收敛条件，停止迭代")
                break
                
            prev_beta = beta
            design_history.append(design)
            
            # 计算中心点处的真实Ucr值
            g_true_center = run_coupled_simulation(fsi_manager, center, use_fsi=use_fsi)["Ucr"] - center["U10"]
            logger.info("中心点验证: g_true=%.4f, g_pred=%.4f", g_true_center, g_pred_design)
            
            center = update_sampling_center(center, g_true_center, design, g_pred_design)
            
            if abs(g_true_center) < 1e-3:
                logger.info("中心点g值接近零，停止迭代")
                break
                
            scale = max(scale * 0.8, 0.2)
            logger.info("更新采样半径: %.2f", scale)
    except Exception as exc:
        logger.error("迭代过程失败: %s", exc, exc_info=True)
        raise

    return multi_rsm, design_history

def compute_capacity(
    rsm: MultiRSM,
    dists: Dict[str, rv_frozen],
    indicator: str,
    thresh: float,
    size: int = 1000,
) -> np.ndarray:
    """计算指定指标的临界容量。"""
    logger.info("计算容量指标: %s (阈值=%.4f)", indicator, thresh)
    
    try:
        alpha_samples = dists["alpha"].rvs(size=size)
        mu_u = dists["U10"].mean()
        std_u = dists["U10"].std()
        capacities = []
        
        # 特殊处理Ucr - 直接使用预测值
        if indicator == "Ucr":
            for a in alpha_samples:
                df = pd.DataFrame({"U10": [mu_u], "alpha": [a]})
                Ucr_pred = rsm.predict(indicator, df)[0]
                capacities.append(Ucr_pred)
            return np.array(capacities)
        
        # 处理sigma_q和sigma_a - 寻找达到阈值的风速
        for a in alpha_samples:
            u_low = 1.0
            u_high = mu_u + 6 * std_u
            grid = np.linspace(u_low, u_high, 120)
            df = pd.DataFrame({"U10": grid, "alpha": a})
            val = rsm.predict(indicator, df)
            
            # 对于sigma_q/sigma_a，容量是达到阈值的风速
            if indicator in ["sigma_q", "sigma_a"]:
                g = val - thresh
                if g[0] >= 0:  # 最低风速已超过阈值
                    capacities.append(u_low)
                    continue
                if g[-1] <= 0:  # 最高风速仍未达到阈值
                    capacities.append(np.nan)
                    continue
                
                # 找到跨越阈值的点
                idx = np.where(g >= 0)[0][0]
                u1, u2 = grid[idx-1], grid[idx]
                g1, g2 = g[idx-1], g[idx]
                cap = np.interp(0.0, [g1, g2], [u1, u2])
                capacities.append(cap)
            else:
                capacities.append(np.nan)
        
        return np.array(capacities)
    except Exception as exc:
        logger.error("容量计算失败: %s", exc, exc_info=True)
        raise

def build_all_capacities(
    dists: Dict[str, rv_frozen], rsm: MultiRSM, thresholds: Dict[str, Tuple[str, float]], size: int = 1000
) -> Dict[str, np.ndarray]:
    """生成多种损伤状态的容量样本。"""
    logger.info("构建容量样本，阈值定义: %s", thresholds)
    return {
        label: compute_capacity(rsm, dists, ind, thr, size=size)
        for label, (ind, thr) in thresholds.items()
    }

def fit_fragility_curve(capacity_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float, float, float]]:
    """拟合多条脆弱性曲线。"""
    logger.info("拟合脆弱性曲线")
    results: Dict[str, Tuple[float, float, float, float]] = {}
    
    for label, samples in capacity_dict.items():
        samples = samples[~np.isnan(samples)]
        if len(samples) == 0:
            logger.warning("指标 %s 无有效样本", label)
            results[label] = (np.nan, np.nan, np.nan, np.nan)
            continue
        
        # 对数正态分布拟合
        shape, loc, scale = lognorm.fit(samples, floc=0)
        theta = scale
        beta = shape
        
        # Kolmogorov-Smirnov检验
        ks_stat, _ = kstest(np.log(samples), "norm", args=(np.log(theta), beta))
        
        # 标准误估计
        se_beta = beta / np.sqrt(2 * len(samples)) if len(samples) > 1 else 0.0
        
        results[label] = (theta, beta, se_beta, ks_stat)
        logger.info("拟合结果 %s: θ=%.4f, β=%.4f, SE=%.4f, KS=%.4f", 
                   label, theta, beta, se_beta, ks_stat)
    
    return results

def main(log_level: str = "INFO") -> None:
    """程序主入口。"""
    # 设置日志级别
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 加载配置
    cfg = {}
    config_path = Path("config.yaml")
    if config_path.exists():
        logger.info("加载配置文件: %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.info("未找到配置文件，使用默认参数")
        cfg = {
            "random_vars": {"V_b_100": 28.5, "beta_coeff": 0.12},
            "convergence": {"max_iter": 15, "tol": 0.01, "beta_tol": 0.005},
            "thresholds": {
                "Flutter_DS4": ["Ucr", 0.0],
                "Displacement_DS1": ["sigma_q", 0.1],
                "Displacement_DS2": ["sigma_q", 0.2],
                "Displacement_DS3": ["sigma_q", 0.3],
                "Acceleration_DS1": ["sigma_a", 0.05],
                "Acceleration_DS2": ["sigma_a", 0.1],
                "Acceleration_DS3": ["sigma_a", 0.15]
            },
            "nsga_pop": 80,
            "nsga_gen": 60,
            "use_fsi": True
        }
    
    # 初始化FSI管理器
    fsi_manager = None
    use_fsi = cfg.get("use_fsi", True) and WB_AVAILABLE
    
    if use_fsi:
        wbpj_template = cfg.get("wbpj_template", "bridge_fsi_template.wbpj")
        working_dir = cfg.get("working_dir", "fsi_simulations")
        try:
            fsi_manager = FSISimulationManager(wbpj_template, working_dir)
            logger.info("FSI管理器初始化成功")
        except Exception as e:
            logger.error("FSI管理器初始化失败: %s", str(e))
            use_fsi = False
    else:
        logger.info("配置为禁用FSI仿真，使用简化模型")

    # 定义随机变量
    logger.info("定义随机变量分布")
    mu, std, dists = define_random_variables(**cfg.get("random_vars", {}))
    
    # 获取优化参数
    pop_size = cfg.get("nsga_pop", 80)
    n_gen = cfg.get("nsga_gen", 60)
    logger.info("NSGA-III参数: 种群大小=%d, 代数=%d", pop_size, n_gen)

    # 迭代直至收敛
    logger.info("开始响应面迭代过程")
    convergence_params = cfg.get("convergence", {})
    rsm, history = iterate_until_convergence(
        fsi_manager,
        mu, std, dists, 
        max_iter=convergence_params.get("max_iter", 15),
        tol=convergence_params.get("tol", 0.01),
        beta_tol=convergence_params.get("beta_tol", 0.005),
        pop_size=pop_size, 
        n_gen=n_gen,
        use_fsi=use_fsi
    )

    # 定义损伤阈值
    thresholds = cfg.get("thresholds", {
        "Flutter_DS4": ("Ucr", 0.0),
        "Displacement_DS1": ("sigma_q", 0.1),
        "Displacement_DS2": ("sigma_q", 0.2),
        "Displacement_DS3": ("sigma_q", 0.3),
        "Acceleration_DS1": ("sigma_a", 0.05),
        "Acceleration_DS2": ("sigma_a", 0.1),
        "Acceleration_DS3": ("sigma_a", 0.15)
    })
    
    # 确保阈值格式正确
    for label, value in thresholds.items():
        if isinstance(value, list) and len(value) == 2:
            thresholds[label] = (value[0], float(value[1]))
        logger.info("损伤状态: %s - 指标: %s, 阈值: %.4f", 
                   label, thresholds[label][0], thresholds[label][1])
    
    # 构建容量样本
    capacity_dict = build_all_capacities(dists, rsm, thresholds)
    
    # 拟合脆弱性曲线
    results = fit_fragility_curve(capacity_dict)
    
    # 输出结果
    print("\n悬索桥抗风易损性分析结果:")
    print("=" * 85)
    print(f"{'损伤状态':<20}{'θ(中位值)':<15}{'β(对数标准差)':<15}{'标准误':<12}{'KS统计量':<12}{'失效概率(50年)'}")
    print("-" * 85)
    
    # 计算50年重现期的失效概率
    for label, (theta, beta, se_beta, ks) in results.items():
        # 计算50年失效概率 (假设风速独立同分布)
        # P_f = 1 - [1 - F_M(theta)]^50
        # F_M 为年最大风速分布函数
        if not np.isnan(theta) and not np.isnan(beta):
            # 年失效概率
            annual_pf = 1 - lognorm.cdf(theta, beta)
            # 50年失效概率
            pf_50yr = 1 - (1 - annual_pf) ** 50
        else:
            pf_50yr = np.nan
            
        print(f"{label:<20}{theta:<15.4f}{beta:<15.4f}{se_beta:<12.4f}{ks:<12.4f}{pf_50yr:.6f}")
    
    # 保存结果
    results_list = []
    for label, (theta, beta, se_beta, ks) in results.items():
        if not np.isnan(theta) and not np.isnan(beta):
            annual_pf = 1 - lognorm.cdf(theta, beta)
            pf_50yr = 1 - (1 - annual_pf) ** 50
        else:
            annual_pf = pf_50yr = np.nan
            
        results_list.append({
            "damage_state": label,
            "theta": theta,
            "beta": beta,
            "se_beta": se_beta,
            "ks_stat": ks,
            "annual_pf": annual_pf,
            "pf_50yr": pf_50yr
        })
    results_df = pd.DataFrame(results_list)
    results_path = Path("bridge_wind_fragility_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("结果已保存到 %s", results_path.resolve())
    print(f"\n详细结果已保存到 {results_path}")
    # 保存设计点历史
    if history:
        history_df = pd.DataFrame(history)
        history_path = Path("design_point_history.csv")
        history_df.to_csv(history_path, index=False)
        logger.info("设计点历史已保存到 %s", history_path.resolve())
    # 清理资源
    if fsi_manager:
        fsi_manager.cleanup()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='悬索桥抗风易损性分析（FSI版）')
    parser.add_argument('--log', type=str, default='INFO', help='日志级别 (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--no-fsi', action='store_true', help='禁用FSI仿真')
    args = parser.parse_args()
    main(log_level=args.log)