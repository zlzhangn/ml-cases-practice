"""客户流失预测案例 — 一键运行入口.

执行完整流程：数据加载 -> 清洗 -> 特征工程 -> 训练 -> 评估 -> SHAP -> MLflow.
"""

import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# 将项目根目录加入路径，以便导入公共库
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from data_prep import clean_data, get_feature_target_split, load_data_from_hf  # noqa: E402
from evaluate import evaluate_model, shap_analysis  # noqa: E402
from features import build_preprocessor, encode_target, identify_column_types  # noqa: E402
from models import build_baseline_model, build_main_model  # noqa: E402

from src.data_utils import load_config  # noqa: E402
from src.experiment import (  # noqa: E402
    log_config,
    log_metrics_from_dict,
    log_model,
    log_params_from_dict,
    setup_mlflow,
    start_run,
)
from src.model_utils import save_metrics, save_model  # noqa: E402

RANDOM_SEED = 42


def setup_logging(config: dict) -> None:
    """配置日志."""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )


def main() -> None:
    """主流程."""
    # 1. 加载配置
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("启动案例: %s", config["project"]["name"])
    logger.info("=" * 50)

    output_dir = Path(__file__).parent.parent / config["paths"]["outputs-dir"].lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载数据
    cache_dir = Path(__file__).parent.parent / config["paths"]["cache-dir"].lstrip("./")
    df_raw = load_data_from_hf(config["data"]["dataset-name"], cache_dir=cache_dir)

    # 3. 数据清洗
    df_clean = clean_data(df_raw, config)

    # 4. 拆分 X / y
    X, y = get_feature_target_split(df_clean, config)
    y = encode_target(y)

    # 5. 训练 / 测试拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test-size"],
        stratify=y if config["data"]["stratify"] else None,
        shuffle=config["data"]["shuffle"],
        random_state=config["project"]["random-seed"],
    )
    logger.info("训练集: %s, 测试集: %s", X_train.shape, X_test.shape)

    # 6. 特征工程（拟合在训练集上）
    col_types = identify_column_types(X_train, config)
    preprocessor = build_preprocessor(col_types, config)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info("预处理后特征数: %d", X_train_processed.shape[1])

    # 保存预处理后的特征名（供后续使用）
    feature_names = list(X_train_processed.columns)

    # 7. 初始化 MLflow
    mlflow_config = config["mlflow"]
    setup_mlflow(
        tracking_uri=mlflow_config.get("tracking-uri"),
        experiment_name=mlflow_config["experiment-name"],
    )

    # 8. 训练并评估基线模型
    with start_run(run_name="baseline_logistic_regression"):
        log_params_from_dict(config["model"]["logistic-regression"])
        log_config(config_path)

        baseline_model = build_baseline_model(config)
        baseline_model.fit(X_train_processed, y_train)

        baseline_metrics = evaluate_model(
            baseline_model,
            X_test_processed,
            y_test,
            model_name="baseline",
            output_dir=output_dir,
        )
        log_metrics_from_dict(baseline_metrics)
        log_model(baseline_model, artifact_path="model")

        save_model(baseline_model, output_dir / "models" / "baseline_model.pkl")
        save_metrics(baseline_metrics, output_dir / "reports" / "baseline_metrics.json")

    # 9. 训练并评估主力模型
    with start_run(run_name="main_lightgbm"):
        log_params_from_dict(config["model"]["lightgbm"])
        log_config(config_path)

        main_model = build_main_model(config)
        main_model.fit(X_train_processed, y_train)

        main_metrics = evaluate_model(
            main_model,
            X_test_processed,
            y_test,
            model_name="lightgbm",
            output_dir=output_dir,
        )
        log_metrics_from_dict(main_metrics)
        log_model(main_model, artifact_path="model")

        save_model(main_model, output_dir / "models" / "lightgbm_model.pkl")
        save_metrics(main_metrics, output_dir / "reports" / "lightgbm_metrics.json")

        # 10. SHAP 可解释性分析
        shap_analysis(
            main_model,
            X_test_processed,
            feature_names=feature_names,
            sample_size=config["evaluation"]["shap-sample-size"],
            output_dir=output_dir,
            model_name="lightgbm",
        )

    # 11. 输出最终结论
    logger.info("=" * 50)
    logger.info("案例运行完成!")
    logger.info(
        "基线模型 (Logistic Regression) - AUC: %.4f, F1: %.4f",
        baseline_metrics.get("auc", 0),
        baseline_metrics["f1"],
    )
    logger.info(
        "主力模型 (LightGBM) - AUC: %.4f, F1: %.4f", main_metrics.get("auc", 0), main_metrics["f1"]
    )
    logger.info("=" * 50)

    # 验证退出标准
    auc_threshold = 0.80
    if main_metrics.get("auc", 0) > auc_threshold:
        logger.info("✅ 退出标准达成: AUC = %.4f > %.2f", main_metrics["auc"], auc_threshold)
    else:
        logger.warning(
            "❌ 退出标准未达成: AUC = %.4f <= %.2f", main_metrics.get("auc", 0), auc_threshold
        )


if __name__ == "__main__":
    main()
