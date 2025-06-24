```mermaid
flowchart TD
  A[CCD 采样] --> B[RSM 拟合 / NSGA-III]
  B --> C[FORM 设计点]
  C -->|更新中心| A
  B --> D[Monte Carlo 容量]
  D --> E[Log-normal 拟合 & 曲线]
