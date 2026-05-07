"""公共工具库：数据加载、缓存、预处理等通用函数."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_config(config_path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件.

    Args:
        config_path: 配置文件路径.

    Returns:
        配置字典.

    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("加载配置: %s", config_path)
    return config


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """保存配置到 YAML 文件.

    Args:
        config: 配置字典.
        config_path: 目标文件路径.

    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    logger.info("保存配置: %s", config_path)


def load_csv(
    file_path: str | Path,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """加载 CSV 文件，支持缓存加速.

    若 cache_dir 指定且存在对应的 parquet 缓存，则直接读取缓存.
    否则读取 CSV 并生成 parquet 缓存（下次加速）.

    Args:
        file_path: CSV 文件路径.
        cache_dir: 缓存目录路径，None 则不缓存.
        **kwargs: 透传给 pd.read_csv 的额外参数.

    Returns:
        DataFrame.

    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 使用文件路径的 hash 作为缓存名，避免文件名冲突
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"{file_path.stem}_{file_hash}.parquet"
        if cache_path.exists():
            logger.info("命中缓存: %s", cache_path)
            return pd.read_parquet(cache_path)

    logger.info("读取 CSV: %s", file_path)
    df = pd.read_csv(file_path, **kwargs)

    if cache_path is not None:
        df.to_parquet(cache_path, index=False)
        logger.info("写入缓存: %s", cache_path)

    return df


def save_csv(df: pd.DataFrame, file_path: str | Path, **kwargs: Any) -> None:
    """保存 DataFrame 到 CSV 文件.

    Args:
        df: 待保存的 DataFrame.
        file_path: 目标文件路径.
        **kwargs: 透传给 to_csv 的额外参数.

    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False, **kwargs)
    logger.info("保存 CSV: %s", file_path)


def split_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """将 DataFrame 拆分为特征矩阵 X 和目标向量 y.

    Args:
        df: 输入数据.
        target_col: 目标列名.
        drop_cols: 额外需要丢弃的列名列表.

    Returns:
        (X, y) 元组.

    """
    drop_cols = drop_cols or []
    cols_to_drop = [target_col] + drop_cols
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col]
    return X, y


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """通过类型降级减少 DataFrame 内存占用.

    对整数和浮点列自动选择最小可容纳类型.

    Args:
        df: 输入 DataFrame.

    Returns:
        内存优化后的 DataFrame.

    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("int8").min and c_max < np.iinfo("int8").max:
                df[col] = df[col].astype("int8")
            elif c_min > np.iinfo("int16").min and c_max < np.iinfo("int16").max:
                df[col] = df[col].astype("int16")
            elif c_min > np.iinfo("int32").min and c_max < np.iinfo("int32").max:
                df[col] = df[col].astype("int32")
        elif pd.api.types.is_float_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("float32").min and c_max < np.finfo("float32").max:
                df[col] = df[col].astype("float32")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        "内存优化: %.2f MB -> %.2f MB (减少 %.1f%%)",
        start_mem,
        end_mem,
        100 * (start_mem - end_mem) / start_mem,
    )
    return df
