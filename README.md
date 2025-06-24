flowchart TD
    %% ---------- 预处理 ----------
    A[开始] --> B[随机变量定义<br/>define_random_variables()]
    B --> C[初始化迭代参数<br/>center=μ, scale=1.0]

    %% ---------- 主循环 ----------
    subgraph LOOP[迭代 until convergence]
        direction TB
        C --> D[CCD 采样<br/>generate_ccd_samples()]
        D --> E[并行流固仿真<br/>run_simulations_async()]
        E --> F[多指标 RSM 拟合<br/>MultiRSM.fit_all()]
        F --> G[NSGA‑III 优化<br/>QuadraticRSM.optimize()]
        G --> H[FORM 计算 β<br/>ReliabilitySolver.solve()]
        H --> I{收敛判据<br/>(Δβ & Δx)}
        I -- 否 --> J[更新中心点<br/>update_sampling_center()]
        J --> K[缩放采样半径<br/>scale × 0.8]
        K --> D
        I -- 是 --> L[退出循环]
    end

    %% ---------- 后处理 ----------
    L --> M[Monte‑Carlo 求容量<br/>compute_capacity()]
    M --> N[对数正态拟合<br/>fit_fragility_curve()]
    N --> O[输出 5 条脆弱性曲线参数]
    O --> P[结束]
