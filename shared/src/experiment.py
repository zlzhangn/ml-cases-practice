"""公共工具库：MLflow 实验管理封装."""

import logging
import uuid
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def setup_mlflow(tracking_uri: str | None = None, experiment_name: str = "default") -> str:
    """初始化 MLflow 追踪环境.

    Args:
        tracking_uri: MLflow tracking URI，None 则使用默认本地文件存储.
        experiment_name: 实验名称.

    Returns:
        实验 ID.

    """
    if tracking_uri is None:
        tracking_uri = f"file://{Path.cwd() / 'mlruns'}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    experiment = mlflow.set_experiment(experiment_name)
    logger.info("设置实验: %s (ID: %s)", experiment_name, experiment.experiment_id)
    return experiment.experiment_id


def start_run(run_name: str | None = None, nested: bool = False) -> Any:
    """启动一个 MLflow Run.

    Args:
        run_name: Run 名称，None 则自动生成.
        nested: 是否作为嵌套 Run.

    Returns:
        ActiveRun 上下文管理器.

    """
    if run_name is None:
        run_name = f"run_{uuid.uuid4().hex[:8]}"
    return mlflow.start_run(run_name=run_name, nested=nested)


def log_params_from_dict(params: dict[str, Any]) -> None:
    """批量记录参数到当前 Run.

    Args:
        params: 参数字典.

    """
    for key, value in params.items():
        # MLflow 只接受标量或短字符串，复杂类型转为字符串
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))
    logger.info("已记录 %d 个参数", len(params))


def log_metrics_from_dict(metrics: dict[str, float], step: int | None = None) -> None:
    """批量记录指标到当前 Run.

    Args:
        metrics: 指标字典，值必须是数值型.
        step: 可选的步数（如训练轮次）.

    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=step)
        else:
            logger.warning("指标 %s 不是数值类型，跳过: %s", key, type(value))
    logger.info("已记录 %d 个指标", len(metrics))


def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """上传本地文件作为 Artifact.

    Args:
        local_path: 本地文件路径.
        artifact_path: Artifact 子目录.

    """
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
    logger.info("已上传 Artifact: %s", local_path)


def log_model(
    model: Any,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> None:
    """记录模型到当前 Run.

    优先使用 sklearn 的 log_model，否则尝试通用方式.

    Args:
        model: 训练好的模型.
        artifact_path: Artifact 路径.
        registered_model_name: 注册模型名称（如需要注册到 Model Registry）.

    """
    try:
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
    except Exception:
        # 若 sklearn 接口不可用，尝试直接 pickle
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            registered_model_name=registered_model_name,
        )
    logger.info("已记录模型到: %s", artifact_path)


def log_config(config_path: str | Path, artifact_path: str = "config") -> None:
    """记录配置文件到当前 Run.

    Args:
        config_path: 配置文件路径.
        artifact_path: Artifact 子目录.

    """
    mlflow.log_artifact(str(config_path), artifact_path=artifact_path)
    logger.info("已记录配置文件: %s", config_path)


def get_best_run(
    experiment_id: str,
    metric_key: str = "f1",
    mode: str = "max",
) -> dict[str, Any] | None:
    """从实验中获取最优 Run.

    Args:
        experiment_id: 实验 ID.
        metric_key: 用于排序的指标名.
        mode: "max" 或 "min".

    Returns:
        最优 Run 的信息字典，若不存在则返回 None.

    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_key} {'DESC' if mode == 'max' else 'ASC'}"],
        max_results=1,
    )
    if not runs:
        return None
    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "metrics": best.data.metrics,
        "params": best.data.params,
    }
