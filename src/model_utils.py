"""公共工具库：模型训练、评估、保存的统一接口."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    **fit_params: Any,
) -> BaseEstimator:
    """训练模型.

    Args:
        model: 未训练的 scikit-learn 兼容模型.
        X_train: 训练特征.
        y_train: 训练标签.
        **fit_params: 透传给 model.fit 的额外参数.

    Returns:
        训练后的模型.

    """
    logger.info("开始训练模型: %s", model.__class__.__name__)
    model.fit(X_train, y_train, **fit_params)
    logger.info("模型训练完成")
    return model


def evaluate_classification(
    model: BaseEstimator,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    proba: bool = True,
) -> dict[str, float]:
    """评估分类模型，返回常用指标.

    Args:
        model: 已训练的分类模型.
        X_test: 测试特征.
        y_test: 测试标签.
        proba: 是否计算基于概率的指标（如 AUC）.

    Returns:
        指标字典，包含 accuracy、precision、recall、f1、auc 等.

    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
    }

    if proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["auc"] = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics["auc_manual"] = auc(fpr, tpr)

    logger.info("分类评估结果: %s", metrics)
    return metrics


def evaluate_regression(
    model: BaseEstimator,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """评估回归模型，返回常用指标.

    Args:
        model: 已训练的回归模型.
        X_test: 测试特征.
        y_test: 测试标签.

    Returns:
        指标字典，包含 rmse、mae、mape、r2 等.

    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    r2 = model.score(X_test, y_test)

    metrics = {
        "rmse": np.sqrt(mse),
        "mae": mae,
        "mape": mape,
        "r2": r2,
    }
    logger.info("回归评估结果: %s", metrics)
    return metrics


def print_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> None:
    """打印分类报告和混淆矩阵.

    Args:
        y_true: 真实标签.
        y_pred: 预测标签.

    """
    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n=== 混淆矩阵 ===")
    print(confusion_matrix(y_true, y_pred))


def save_model(model: BaseEstimator, path: str | Path) -> None:
    """保存模型到磁盘.

    优先使用 joblib，回退到 pickle.

    Args:
        model: 待保存的模型.
        path: 保存路径.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    logger.info("模型已保存: %s", path)


def load_model(path: str | Path) -> Any:
    """从磁盘加载模型.

    Args:
        path: 模型文件路径.

    Returns:
        加载后的模型对象.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")

    try:
        model = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            model = pickle.load(f)
    logger.info("模型已加载: %s", path)
    return model


def save_metrics(metrics: dict[str, float], path: str | Path) -> None:
    """保存指标到 JSON 文件.

    Args:
        metrics: 指标字典.
        path: 保存路径.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("指标已保存: %s", path)
