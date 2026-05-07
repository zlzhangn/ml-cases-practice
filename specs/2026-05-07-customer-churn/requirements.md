# 客户流失预测 — 需求规格

## 背景

客户流失（Churn）是电信、金融、SaaS 等订阅制业务中的核心问题。准确预测哪些客户即将流失，可以帮助业务侧提前介入挽留，降低获客成本。本项目选择 HuggingFace 上的电信客户流失公开数据集，作为第一个端到端案例，帮助学习者建立完整的 ML 实战流程。

## 功能范围

### 包含

- 从 HuggingFace 自动加载并缓存 `aai510-group1/telco-customer-churn` 数据集
- 完整的数据清洗、EDA、特征工程流程
- 基线模型（Logistic Regression）+ 主力模型（LightGBM）的训练与对比
- 基于 AUC / F1 / Confusion Matrix 的模型评估
- SHAP 可解释性分析，输出 Top 5 关键特征及其业务含义
- MLflow 实验追踪，记录首个可复现实验

### 不包含

- 深度学习模型（神经网络、Transformer 等），依据 `mission.md` 范围限定
- 模型部署与服务化（仅做实验追踪，不做 CI/CD / API 服务）
- 实时流式预测系统
- 多语言文本处理（本数据集为结构化表格数据）

## 关键决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 主力模型 | LightGBM | 表格数据场景下性能优异，训练速度快，与 scikit-learn 兼容 |
| 基线模型 | Logistic Regression | 可解释性强，作为性能下限参考 |
| 可解释性工具 | SHAP | 与 LightGBM 集成成熟，支持全局 + 局部解释 |
| 实验追踪 | MLflow | 项目统一标准，本地文件系统存储，零运维成本 |
| 特征编码 | 混合策略 | 低基数类别用 One-Hot，高基数用 Label Encoding |
| 评估指标 | AUC + F1 | AUC 对阈值不敏感，F1 平衡 Precision / Recall |

## 约束条件

- 技术栈：Python 3.11+，uv 包管理，ruff 格式化
- 代码风格：中文注释/文档，最大行宽 100，函数必须有 docstring
- 目录结构：严格遵循 `01-customer-churn/` 标准模板
- 随机种子：固定为 `RANDOM_SEED = 42`，确保结果可复现

## 依赖数据

- **数据集**：`aai510-group1/telco-customer-churn`（HuggingFace Datasets）
- **数据说明**：电信客户信息（人口统计、服务订阅、账户信息、流失标记）
- **目标变量**：`Churn`（Yes / No）

## 参考文档

- `specs/mission.md` — 项目使命与范围限定
- `specs/tech-stack.md` — 技术选型与目录规范
- `specs/roadmap.md` — Phase 1 原始任务清单
