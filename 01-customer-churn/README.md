# 案例 1：客户流失预测（Customer Churn Prediction）

## 简介

电信行业客户流失预测是机器学习分类任务的经典入门案例。本案例使用 HuggingFace 上的公开数据集，建立端到端的客户流失预测流程。

## 数据集

- **来源**：HuggingFace Datasets — `aai510-group1/telco-customer-churn`
- **规模**：约 7,043 行 × 21 列
- **目标变量**：`Churn`（Yes / No）
- **特征类型**：客户人口统计信息、订阅服务、账户信息

## 运行方式

### 环境准备

```bash
# 在项目根目录执行
uv sync
```

### 一键运行

```bash
cd 01-customer-churn
uv run python src/main.py
```

运行完成后：
- 模型和指标保存在 `outputs/` 目录
- 图表保存在 `outputs/figures/` 目录
- MLflow 实验记录保存在 `mlruns/` 目录

### 查看实验记录

```bash
uv run python -m mlflow ui --backend-store-uri ./mlruns
```

浏览器访问 `http://localhost:5000` 查看实验对比。

## 核心流程

1. **数据加载**：从 HuggingFace 自动下载数据集
2. **数据清洗**：处理缺失值、统一编码格式
3. **特征工程**：数值标准化 + 类别 One-Hot / Ordinal 编码
4. **模型训练**：
   - 基线模型：Logistic Regression（带类别权重平衡）
   - 主力模型：LightGBM（带类别权重平衡）
5. **模型评估**：AUC、F1、Precision、Recall、Confusion Matrix、ROC 曲线
6. **可解释性**：SHAP 分析，输出 Top 5 关键特征
7. **实验追踪**：MLflow 记录参数、指标、模型

## 核心结论

| 模型 | AUC | F1-Score |
|------|-----|----------|
| Logistic Regression | ~0.82 | ~0.58 |
| LightGBM | ~0.84 | ~0.63 |

> 实际数值以运行结果为准。LightGBM 通常在树型特征和交互效应上表现更优。

## 目录结构

```
01-customer-churn/
├── README.md
├── config.yaml          # 案例专属配置
├── data/                # 数据缓存（不提交 Git）
├── notebooks/           # Jupyter 笔记本（EDA 探索）
├── src/                 # 源码
│   ├── __init__.py
│   ├── data_prep.py     # 数据加载与清洗
│   ├── features.py      # 特征工程
│   ├── models.py        # 模型定义
│   ├── evaluate.py      # 评估与可视化
│   └── main.py          # 一键运行入口
├── mlruns/              # MLflow 实验记录
└── outputs/             # 输出
    ├── models/          # 序列化模型
    ├── figures/         # 图表
    └── reports/         # 指标报告
```

## 退出标准

- ✅ AUC > 0.80
- ✅ 能解释 Top 5 关键特征的业务含义
- ✅ MLflow 实验记录完整可复现

## 参考

- `specs/mission.md` — 项目使命
- `specs/tech-stack.md` — 技术栈规范
- `specs/roadmap.md` — Phase 1 任务清单
