"""模型定义与训练模块.

包含基线模型（Logistic Regression）和主力模型（LightGBM）的构建与训练.
"""

import logging
from typing import Any

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def build_baseline_model(config: dict[str, Any]) -> LogisticRegression:
    """构建逻辑回归基线模型.

    Args:
        config: 配置字典，需包含 model.logistic-regression 字段.

    Returns:
        未训练的逻辑回归模型.

    """
    params = config["model"]["logistic-regression"]
    model = LogisticRegression(**params)
    logger.info("构建基线模型: LogisticRegression(%s)", params)
    return model


def build_main_model(config: dict[str, Any]) -> lgb.LGBMClassifier:
    """构建 LightGBM 主力模型.

    Args:
        config: 配置字典，需包含 model.lightgbm 字段.

    Returns:
        未训练的 LightGBM 模型.

    """
    params = config["model"]["lightgbm"]
    model = lgb.LGBMClassifier(**params)
    logger.info("构建主力模型: LGBMClassifier(%s)", params)
    return model
