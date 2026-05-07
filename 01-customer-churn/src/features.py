"""特征工程模块.

负责类别编码、数值标准化、特征筛选等操作.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def identify_column_types(X: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    """自动识别列类型：数值列、低基数类别列、高基数类别列.

    Args:
        X: 特征矩阵.
        config: 配置字典.

    Returns:
        包含 numeric, low_cardinality, high_cardinality 列名的字典.

    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # 从类别列中排除已经是数值的列（如 Churn、SeniorCitizen 等）
    category_cols = [c for c in X.columns if c not in numeric_cols]

    onehot_max = config["features"]["onehot-max-categories"]

    low_cardinality = []
    high_cardinality = []

    for col in category_cols:
        n_unique = X[col].nunique(dropna=False)
        if n_unique <= onehot_max:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    logger.info(
        "列类型识别完成: 数值=%d, 低基数类别=%d, 高基数类别=%d",
        len(numeric_cols),
        len(low_cardinality),
        len(high_cardinality),
    )

    return {
        "numeric": numeric_cols,
        "low_cardinality": low_cardinality,
        "high_cardinality": high_cardinality,
    }


def build_preprocessor(
    col_types: dict[str, list[str]], config: dict[str, Any]
) -> ColumnTransformer:
    """构建 sklearn ColumnTransformer 预处理管道.

    Args:
        col_types: 列类型字典.
        config: 配置字典.

    Returns:
        ColumnTransformer 对象.

    """
    transformers = []

    # 数值特征标准化
    if col_types["numeric"] and config["features"]["scale-numeric"]:
        transformers.append(("num", StandardScaler(), col_types["numeric"]))
        logger.info("数值特征标准化: %s", col_types["numeric"])
    elif col_types["numeric"]:
        transformers.append(("num", "passthrough", col_types["numeric"]))

    # 低基数类别 One-Hot 编码
    if col_types["low_cardinality"]:
        transformers.append(
            (
                "cat_low",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                col_types["low_cardinality"],
            )
        )
        logger.info("低基数类别 One-Hot 编码: %s", col_types["low_cardinality"])

    # 高基数类别 Label 编码（此处直接映射为整数）
    # 注意：LabelEncoder 通常用于 y，对 X 使用需谨慎；
    # 但对于树模型（LightGBM），整数编码的类别特征是可以接受的
    if col_types["high_cardinality"]:
        # 使用 OrdinalEncoder 更规范
        from sklearn.preprocessing import OrdinalEncoder

        transformers.append(
            (
                "cat_high",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                col_types["high_cardinality"],
            )
        )
        logger.info("高基数类别 Ordinal 编码: %s", col_types["high_cardinality"])

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor


def encode_target(y: pd.Series) -> pd.Series:
    """确保目标变量为数值类型.

    Args:
        y: 原始目标向量.

    Returns:
        数值化的目标向量.

    """
    if y.dtype == object or y.dtype.name == "string":
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
        logger.info("目标变量编码映射: %s", dict(zip(le.classes_, le.transform(le.classes_))))
        return y_encoded
    return y
