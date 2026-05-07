"""公共工具库：常用可视化图表模板."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

# 统一风格
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def save_or_show(fig: plt.Figure, path: str | Path | None = None) -> None:
    """保存或显示图表.

    Args:
        fig: matplotlib Figure 对象.
        path: 保存路径，None 则直接显示.

    """
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info("图表已保存: %s", path)
        plt.close(fig)
    else:
        plt.show()


def plot_corr_heatmap(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (12, 10),
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制特征相关性热力图.

    Args:
        df: 数值型 DataFrame.
        figsize: 图像尺寸.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("特征相关性热力图")
    save_or_show(fig, path)
    return fig


def plot_distribution(
    series: pd.Series,
    title: str = "分布图",
    bins: int = 30,
    kde: bool = True,
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制单变量分布图.

    Args:
        series: 输入序列.
        title: 图表标题.
        bins: 直方图分箱数.
        kde: 是否叠加核密度估计.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, bins=bins, kde=kde, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("频次")
    save_or_show(fig, path)
    return fig


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC 曲线",
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制 ROC 曲线.

    Args:
        y_true: 真实标签.
        y_proba: 预测正类概率.
        title: 图表标题.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="随机猜测")
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    save_or_show(fig, path)
    return fig


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[str] | None = None,
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制混淆矩阵热力图.

    Args:
        y_true: 真实标签.
        y_pred: 预测标签.
        labels: 类别名称.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    tick_labels = labels if labels is not None else "auto"
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵")
    save_or_show(fig, path)
    return fig


def plot_feature_importance(
    importances: pd.Series | dict[str, float] | np.ndarray,
    feature_names: list[str] | None = None,
    top_n: int = 20,
    title: str = "特征重要性",
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制特征重要性条形图.

    Args:
        importances: 特征重要性值.
        feature_names: 特征名称列表.
        top_n: 显示前 N 个特征.
        title: 图表标题.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    if isinstance(importances, dict):
        importances = pd.Series(importances)
    elif isinstance(importances, np.ndarray):
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        importances = pd.Series(importances, index=feature_names)

    importances = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("重要性")
    save_or_show(fig, path)
    return fig


def plot_time_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    title: str = "时间序列",
    path: str | Path | None = None,
) -> plt.Figure:
    """绘制时间序列折线图.

    Args:
        df: 包含时间序列数据的 DataFrame.
        time_col: 时间列名.
        value_col: 数值列名.
        title: 图表标题.
        path: 保存路径.

    Returns:
        Figure 对象.

    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[time_col], df[value_col], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(time_col)
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    save_or_show(fig, path)
    return fig
