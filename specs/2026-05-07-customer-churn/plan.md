# 客户流失预测 — 开发计划

> 对应路线图 Phase 1，目标：跑通第一个完整案例，建立信心。

---

## 任务组 1：数据准备

1.1 从 HuggingFace 加载数据集 `aai510-group1/telco-customer-churn`
1.2 数据清洗（缺失值、异常值、格式统一）
1.3 初步了解数据规模、字段含义、目标变量分布

## 任务组 2：探索性数据分析（EDA）

2.1 单变量分析（数值/类别变量分布）
2.2 目标变量与特征的关联分析（交叉表、分组统计）
2.3 可视化关键发现（seaborn / matplotlib）

## 任务组 3：特征工程

3.1 类别特征编码（One-Hot / Label Encoding）
3.2 数值特征标准化 / 归一化
3.3 特征筛选（基于相关性、业务含义）

## 任务组 4：模型训练

4.1 划分训练集 / 测试集（固定随机种子，防止数据泄漏）
4.2 基线模型：Logistic Regression
4.3 主力模型：LightGBM（严格限定传统 ML 方法，不使用深度学习）
4.4 超参数调优（Optuna 或 Grid Search）

## 任务组 5：模型评估

5.1 核心指标：AUC、F1-Score
5.2 辅助指标：Confusion Matrix、Precision、Recall
5.3 结果可视化（ROC 曲线、PR 曲线）

## 任务组 6：可解释性分析

6.1 SHAP 值计算与可视化
6.2 提取 Top 5 关键特征
6.3 用业务语言解释特征对流失的影响方向

## 任务组 7：实验追踪

7.1 配置 MLflow Experiment（命名为 `01-customer-churn`）
7.2 记录参数（`config.yaml` 内容）、指标、模型文件
7.3 验证 MLflow UI 可正确查看实验记录

---

## 备注

- 所有代码遵循 `tech-stack.md` 规范（ruff 格式化、中文注释/文档、最大行宽 100）
- 案例目录结构遵循 `0X-case-name/` 标准模板
