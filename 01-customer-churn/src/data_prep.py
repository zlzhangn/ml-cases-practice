"""数据加载与预处理模块.

负责从 HuggingFace 或 OpenML 下载电信客户流失数据集，并进行清洗和初步处理.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_data_from_hf(dataset_name: str, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """加载数据集并转换为 pandas DataFrame.

    优先尝试 HuggingFace，若不可访问则自动回退到 OpenML (data_id=42178).

    Args:
        dataset_name: HuggingFace 数据集名称，如 "aai510-group1/telco-customer-churn".
        cache_dir: 本地缓存目录.

    Returns:
        清洗前的原始数据 DataFrame.

    """
    # 先快速检测 HuggingFace 连通性
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://huggingface.co",
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        logger.warning("HuggingFace 网络不可达，直接使用 OpenML 数据源")
        data = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        df = data.frame
        logger.info("OpenML 数据集加载完成: %d 行 x %d 列", len(df), len(df.columns))
        return df

    logger.info("从 HuggingFace 加载数据集: %s", dataset_name)

    try:
        ds = load_dataset(
            dataset_name,
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None,
            download_mode="reuse_cache_if_exists",
        )
        df = ds.to_pandas()
        logger.info("数据集加载完成: %d 行 x %d 列", len(df), len(df.columns))
        return df
    except Exception as e:
        logger.warning("HuggingFace 加载失败: %s", e)
        logger.info("回退到 OpenML 加载数据集 (data_id=42178)...")
        data = fetch_openml(data_id=42178, as_frame=True, parser="auto")
        df = data.frame
        logger.info("OpenML 数据集加载完成: %d 行 x %d 列", len(df), len(df.columns))
        return df


def clean_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """清洗原始数据.

    处理内容包括：
    - 去除客户 ID 列（无预测价值）
    - 处理 TotalCharges 等数值列中的空格/空字符串
    - 统一二元类别映射为 0/1
    - 处理缺失值

    Args:
        df: 原始数据.
        config: 配置字典，需包含 preprocessing 相关字段.

    Returns:
        清洗后的 DataFrame.

    """
    df = df.copy()

    # 1. 去除 ID 列
    id_col = config["data"]["id-col"]
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        logger.info("去除 ID 列: %s", id_col)

    # 2. 处理数值列中的特殊缺失值（空格字符串）
    numeric_like_cols = config["preprocessing"]["numeric-like-cols"]
    placeholder = config["preprocessing"]["missing-placeholder"]

    for col in numeric_like_cols:
        if col in df.columns:
            # 将空格字符串替换为 NaN
            df[col] = df[col].replace(placeholder, np.nan)
            # 转换为数值类型
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # 对于 TotalCharges，缺失值用 0 填充（新用户尚未产生总费用）
            if col == "TotalCharges":
                na_count = df[col].isna().sum()
                if na_count > 0:
                    df[col] = df[col].fillna(0)
                    logger.info("列 %s: %d 个缺失值已填充为 0", col, na_count)

    # 3. 统一二元类别映射
    binary_cols = config["preprocessing"]["binary-cols"]
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
            # 检查是否有未映射的值
            unmapped = df[col].isna().sum()
            if unmapped > 0:
                logger.warning("列 %s: %d 个值未能映射", col, unmapped)

    # 4. 检查并记录缺失值情况
    na_summary = df.isna().sum()
    na_cols = na_summary[na_summary > 0]
    if len(na_cols) > 0:
        logger.warning("清洗后仍存在缺失值的列:\n%s", na_cols)
    else:
        logger.info("清洗后无缺失值")

    logger.info("数据清洗完成: %d 行 x %d 列", len(df), len(df.columns))
    return df


def get_feature_target_split(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """将数据拆分为特征矩阵 X 和目标向量 y.

    Args:
        df: 清洗后的数据.
        config: 配置字典.

    Returns:
        (X, y) 元组.

    """
    target_col = config["data"]["target-col"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info("特征矩阵: %s, 目标向量: %s", X.shape, y.shape)
    return X, y
