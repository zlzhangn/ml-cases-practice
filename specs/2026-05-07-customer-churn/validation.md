# 客户流失预测 — 验证标准

## 代码实现成功的判定标准

### 硬性指标

1. **模型性能**：主力模型（LightGBM）在测试集上的 **AUC > 0.80**
2. **可解释性**：通过 SHAP 分析输出 **Top 5 关键特征**，并能用业务语言解释其对客户流失的影响方向
3. **实验记录**：MLflow 中成功记录至少一个完整 Run，包含参数、指标、模型文件

### 工程标准

4. **目录规范**：案例目录 `01-customer-churn/` 结构完整，包含 `README.md`、`config.yaml`、`notebooks/`、`src/`、`outputs/`、`mlruns/`
5. **配置驱动**：所有可变参数（数据路径、模型参数、评估阈值等）抽取到 `config.yaml`
6. **代码质量**：通过 `ruff check .` 和 `ruff format --check .`，无格式错误
7. **可运行性**：`src/main.py` 可一键运行，从数据加载到模型评估完整走通

## 可以进行合并的条件

- [ ] 硬性指标 1~3 全部满足
- [ ] 工程标准 4~7 全部满足
- [ ] 代码审查通过（self-review 或他人 review）
- [ ] `feature/01-customer-churn` 分支基于最新 `main` 分支，无冲突

## 验证步骤

1. 在干净环境中执行 `uv sync` 安装依赖
2. 运行 `python 01-customer-churn/src/main.py`
3. 检查控制台输出的 AUC / F1 指标
4. 启动 MLflow UI：`mlflow ui --backend-store-uri file://./01-customer-churn/mlruns`，确认实验记录可见
5. 查看 `outputs/figures/` 中的 SHAP 摘要图和 ROC 曲线
6. 运行 `ruff check 01-customer-churn/` 和 `ruff format --check 01-customer-churn/`

## 风险与降级方案

| 风险 | 应对措施 |
|------|----------|
| AUC 达不到 0.80 | 优先检查特征工程和数据清洗，尝试代价敏感学习或类别权重调整 |
| 数据集无法从 HuggingFace 加载 | 使用本地缓存或 Kaggle 镜像作为备选来源 |
| SHAP 计算过慢 | 对大规模数据采样解释子集，确保在合理时间内完成 |

---

*本验证标准与 `specs/roadmap.md` Phase 1 退出标准保持一致。*
