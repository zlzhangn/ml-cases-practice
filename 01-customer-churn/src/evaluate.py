"""评估与可视化模块.

负责模型评估指标计算、图表生成和 SHAP 可解释性分析.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve

from shared.src.model_utils import evaluate_classification, print_classification_report
from shared.src.viz_utils import plot_confusion_matrix, plot_roc_curve, save_or_show

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    output_dir: str | Path = "./outputs",
) -> dict[str, float]:
    """评估模型并保存结果.

    Args:
        model: 已训练的模型.
        X_test: 测试特征.
        y_test: 测试标签.
        model_name: 模型名称（用于日志和文件名）.
        output_dir: 输出目录.

    Returns:
        评估指标字典.

    """
    logger.info("评估模型: %s", model_name)

    # 基础指标
    metrics = evaluate_classification(model, X_test, y_test, proba=True)

    # 预测值
    y_pred = model.predict(X_test)

    # 打印分类报告
    print_classification_report(y_test, y_pred)

    # 保存混淆矩阵图
    cm_path = Path(output_dir) / "figures" / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(y_test, y_pred, labels=["未流失", "流失"], path=cm_path)

    # 保存 ROC 曲线
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_path = Path(output_dir) / "figures" / f"roc_curve_{model_name}.png"
        plot_roc_curve(y_test, y_proba, title=f"{model_name} - ROC 曲线", path=roc_path)

        # 保存 PR 曲线
        pr_path = Path(output_dir) / "figures" / f"pr_curve_{model_name}.png"
        plot_pr_curve(y_test, y_proba, title=f"{model_name} - PR 曲线", path=pr_path)

    logger.info("%s 评估完成: AUC=%.4f, F1=%.4f", model_name, metrics.get("auc", 0), metrics["f1"])
    return metrics


def plot_pr_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    title: str = "PR 曲线",
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制 Precision-Recall 曲线.

    Args:
        y_true: 真实标签.
        y_proba: 预测正类概率.
        title: 图表标题.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("召回率 (Recall)")
    ax.set_ylabel("精确率 (Precision)")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    save_or_show(fig, path)
    return fig


def shap_analysis(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str] | None = None,
    sample_size: int = 500,
    output_dir: str | Path = "./outputs",
    model_name: str = "model",
) -> pd.Series:
    """SHAP 可解释性分析.

    Args:
        model: 已训练的模型（需支持树模型解释器）.
        X_test: 测试特征.
        feature_names: 特征名称列表.
        sample_size: SHAP 计算的采样大小（减少计算时间）.
        output_dir: 输出目录.
        model_name: 模型名称.

    Returns:
        特征重要性 Series（按 SHAP 绝对值均值排序）.

    """
    logger.info("开始 SHAP 分析: %s", model_name)

    # 采样以加速计算
    if len(X_test) > sample_size:
        X_sample = X_test.sample(n=sample_size, random_state=RANDOM_SEED)
    else:
        X_sample = X_test

    # 使用 TreeExplainer（适用于 LightGBM / XGBoost）
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # 对于二分类，shap_values 可能是列表，取正类（索引 1）
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        # 计算特征重要性（SHAP 绝对值均值）
        feature_importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X_sample.columns,
        ).sort_values(ascending=False)

        # 保存 SHAP 摘要图
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"{model_name} - SHAP 特征重要性")
        shap_path = Path(output_dir) / "figures" / f"shap_summary_{model_name}.png"
        save_or_show(fig, shap_path)

        # 保存 Top 5 特征文本说明
        top5 = feature_importance.head(5)
        report_path = Path(output_dir) / "reports" / f"shap_top5_{model_name}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== {model_name} Top 5 关键特征 ===\n\n")
            for feat, importance in top5.items():
                f.write(f"{feat}: {importance:.4f}\n")

        logger.info("SHAP 分析完成，Top 5 特征:\n%s", top5)
        return feature_importance

    except Exception as e:
        logger.error("SHAP 分析失败: %s", e)
        raise
