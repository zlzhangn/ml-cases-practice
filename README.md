# ml-cases-practice

通过真实案例掌握机器学习实战能力。

## 项目简介

本项目通过 7 个递进式真实案例，覆盖分类、回归、时间序列、异常检测、NLP、因果推断、量化回测等核心机器学习领域，帮助学习者建立端到端的 ML 解决能力。

## 环境搭建
windows 用户建议使用 WSL。

```bash
# 安装依赖
uv sync

# 运行验证脚本
python scripts/verify_setup.py
```

## 项目结构

```
ml-cases-practice/
├── specs/              # 项目章程与技术文档
├── src/                # 公共工具库
├── configs/            # 全局配置模板
├── 01-customer-churn/  # 案例 1：客户流失预测
├── 02-sales-forecast/  # 案例 2：销售/需求预测
├── 03-quant-factors/   # 案例 3：量化因子与择时信号
├── 04-anomaly-fraud/   # 案例 4：异常检测与风控
├── 05-nlp-classify/    # 案例 5：NLP 文本分类
├── 06-causal-infer/    # 案例 6：因果推断与策略评估
├── 07-backtest/        # 案例 7：回测框架与 ML 策略工程化
├── pyproject.toml      # 项目依赖与工具配置
├── uv.lock             # 依赖锁定文件
└── README.md           # 项目总览
```

## 快速开始

每个案例目录包含独立的 `README.md`、`config.yaml` 和 `src/main.py`，进入对应目录后运行：

```bash
python src/main.py
```

## 技术栈

- Python 3.11+
- uv（包管理器）
- pandas, numpy, polars（数据处理）
- scikit-learn, LightGBM, XGBoost（机器学习）
- SHAP, MLflow, Optuna（可解释性与实验追踪）
- matplotlib, seaborn, plotly（可视化）

## 开发规范

- 使用 `ruff` 进行代码格式化和检查
- 使用 `uv` 管理依赖
- 使用中文编写所有文档和注释
- 每个案例使用 `config.yaml` 驱动配置
