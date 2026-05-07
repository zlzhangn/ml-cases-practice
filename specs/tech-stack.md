# 技术栈（Tech Stack）

## 环境基座

| 组件 | 选型 | 版本 | 说明 |
|------|------|------|------|
| 编程语言 | Python | 3.11+ | 当前稳定版，所有库均支持 |
| 包管理器 | uv | 最新版 | Rust 实现，速度比 pip/conda 快 10-100x，原生 lock 文件 |
| 虚拟环境 | uv venv | - | 与 uv 集成，无需单独安装 venv/conda |

## 核心依赖

| 库 | 用途 | 版本约束 |
|----|------|---------|
| pandas | 结构化数据处理 | >=2.0 |
| numpy | 数值计算 | >=1.24 |
| polars | 高性能数据处理（备用） | >=0.20 |
| scikit-learn | 基础 ML 算法、预处理、评估、Pipeline | >=1.3 |
| lightgbm | 梯度提升（主力模型） | >=4.0 |
| xgboost | 梯度提升（备选/对比） | >=2.0 |
| shap | 模型可解释性 | >=0.44 |
| mlflow | 实验追踪与模型管理 | >=2.10 |
| optuna | 超参数优化 | >=3.4 |
| matplotlib | 静态可视化 | >=3.7 |
| seaborn | 统计可视化 | >=0.13 |
| plotly | 交互式可视化（EDA 阶段） | >=5.18 |
| pyyaml | YAML 配置文件解析 | >=6.0 |

## 专项依赖（按需安装）

| 库 | 案例 | 用途 |
|----|------|------|
| prophet | 02 | 时间序列基线模型 |
| statsmodels | 02, 03 | 统计检验、ARIMA |
| pyod | 04 | 异常检测算法集合（Isolation Forest, LOF 等） |
| imbalanced-learn | 01, 04 | 不平衡数据采样（SMOTE, ADASYN 等） |
| feature-engine | 全案例 | 高级特征工程（防止数据泄漏） |
| econml | 06 | 因果推断（Double ML, Causal Forest） |
| transformers | 05 | NLP 词向量/预训练模型（按需，案例 5 以传统方法为主） |
| sentence-transformers | 05 | 语义嵌入（备用） |
| jupyter | 全案例 | 交互式开发 |
| notebook | 全案例 | Jupyter Notebook 支持 |

## 开发工具

| 工具 | 用途 | 配置 |
|------|------|------|
| ruff | Python 代码 lint + format | 配置于 `pyproject.toml` |
| VS Code | 主力 IDE | 推荐安装 Python、Jupyter、YAML 插件 |

## 目录与命名规范

### 项目根目录结构

```
opencode-examples/
├── specs/              # 项目章程与技术文档
│   ├── mission.md
│   ├── tech-stack.md
│   └── roadmap.md
├── src/                # 公共工具库
│   ├── __init__.py
│   ├── data_utils.py      # 数据加载、缓存、预处理公共函数
│   ├── model_utils.py     # 模型训练、评估、保存公共函数
│   ├── viz_utils.py       # 可视化公共函数
│   └── experiment.py      # MLflow 实验管理封装
├── configs/            # 全局配置模板
│   └── base_config.yaml
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

### 案例目录结构（每个案例统一）

```
0X-case-name/
├── README.md           # 案例说明、运行指南、核心结论
├── config.yaml         # 案例专属配置（数据路径、模型参数、评估阈值等）
├── data/               # 数据目录（原始数据自动下载缓存，不提交 Git）
│   └── .gitkeep
├── notebooks/          # Jupyter 笔记本（EDA、探索性分析）
│   ├── 01-eda.ipynb
│   └── 02-modeling.ipynb
├── src/                # 案例专属源码
│   ├── __init__.py
│   ├── data_prep.py    # 数据加载与预处理
│   ├── features.py     # 特征工程
│   ├── models.py       # 模型定义与训练
│   ├── evaluate.py     # 评估与可视化
│   └── main.py         # 一键运行入口
├── mlruns/             # MLflow 实验记录（不提交 Git）
│   └── .gitkeep
└── outputs/            # 输出目录（模型、图表、报告，不提交 Git）
    ├── models/
    ├── figures/
    └── reports/
```

### 命名规范

- **目录名**：`0X-英文小写`，如 `01-customer-churn`
- **Python 文件**：小写 + 下划线，如 `data_prep.py`
- **类名**：大驼峰，如 `DataLoader`
- **函数/变量**：小写 + 下划线，如 `load_dataset`
- **常量**：全大写 + 下划线，如 `RANDOM_SEED = 42`
- **配置文件**：`config.yaml`，使用小写 + 连字符的 key，如 `test-size: 0.2`

## 依赖管理

1. **初始化**：`uv init` 创建项目，`uv add <package>` 添加依赖
2. **锁定**：`uv lock` 生成 `uv.lock`，确保环境可复现
3. **安装**：`uv sync` 安装所有依赖
4. **案例专属依赖**：若某案例需要额外库，在案例 `README.md` 中注明，必要时创建 `requirements-case.txt`

## 实验追踪规范

- 每个案例对应 MLflow 中的一个 **Experiment**，命名为 `0X-case-name`
- 每次运行对应一个 **Run**，自动记录：
  - 参数：`config.yaml` 的全部内容
  - 指标：评估指标（AUC、RMSE、Precision、Recall 等）
  - 模型：序列化后的模型文件
  - 标签：数据版本、代码 commit hash（可选）
- MLflow tracking URI 设为本地文件系统：`mlruns/`

## 代码风格

- 使用 **ruff** 统一格式化，配置于 `pyproject.toml`
- 最大行宽：100 字符
- 函数必须有 docstring（中文或英文均可）
- 所有 `print` 用于临时调试，最终输出使用 `logging`
- 不要重复造轮子，优先使用项目根目录`src/` 中的公共函数

---

*本文件为技术决策的权威参考。如需变更技术栈，须更新本文件并记录变更原因。*
