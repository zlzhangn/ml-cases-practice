"""验证脚本：测试基础设施搭建是否完整.

运行方式:
    uv run python scripts/verify_setup.py

"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_setup")


def test_imports() -> bool:
    """测试 src 下所有模块能否正常导入."""
    logger.info("[1/5] 测试模块导入...")
    try:

        logger.info("✓ 所有模块导入成功")
        return True
    except Exception as e:
        logger.error("✗ 模块导入失败: %s", e)
        return False


def test_data_utils() -> bool:
    """测试数据工具函数."""
    logger.info("[2/5] 测试数据工具...")
    try:
        from src.data_utils import (
            load_config,
            reduce_memory_usage,
            save_csv,
            split_xy,
        )

        # 生成测试数据
        df = pd.DataFrame(
            {
                "feature_a": np.random.randn(100),
                "feature_b": np.random.randint(0, 10, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        # 测试 split_xy
        X, y = split_xy(df, target_col="target")
        assert X.shape == (100, 2)
        assert y.shape == (100,)

        # 测试 reduce_memory_usage
        df_optimized = reduce_memory_usage(df.copy())
        assert df_optimized.shape == df.shape

        # 测试 save_csv / load_config
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            save_csv(df, csv_path)
            assert csv_path.exists()

            config = {"test": "value", "number": 42}
            config_path = Path(tmpdir) / "config.yaml"
            from src.data_utils import save_config

            save_config(config, config_path)
            loaded_config = load_config(config_path)
            assert loaded_config["test"] == "value"
            assert loaded_config["number"] == 42

        logger.info("✓ 数据工具测试通过")
        return True
    except Exception as e:
        logger.error("✗ 数据工具测试失败: %s", e)
        return False


def test_model_utils() -> bool:
    """测试模型工具函数."""
    logger.info("[3/5] 测试模型工具...")
    try:
        from src.model_utils import (
            evaluate_classification,
            load_model,
            save_metrics,
            save_model,
            train_model,
        )

        # 生成分类数据
        X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 训练模型
        model = LogisticRegression(max_iter=1000, random_state=42)
        model = train_model(model, X_train, y_train)

        # 评估
        metrics = evaluate_classification(model, X_test, y_test)
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert metrics["accuracy"] > 0.5  # 至少比随机好

        # 保存/加载模型
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            save_model(model, model_path)
            assert model_path.exists()

            loaded = load_model(model_path)
            assert loaded is not None

            metrics_path = Path(tmpdir) / "metrics.json"
            save_metrics(metrics, metrics_path)
            assert metrics_path.exists()

        logger.info("✓ 模型工具测试通过")
        return True
    except Exception as e:
        logger.error("✗ 模型工具测试失败: %s", e)
        return False


def test_viz_utils() -> bool:
    """测试可视化工具函数."""
    logger.info("[4/5] 测试可视化工具...")
    try:
        from src.viz_utils import (
            plot_confusion_matrix,
            plot_distribution,
            plot_roc_curve,
        )

        y_true = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100)
        y_pred = (y_proba > 0.5).astype(int)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试分布图
            fig = plot_distribution(
                pd.Series(y_proba, name="score"),
                path=Path(tmpdir) / "dist.png",
            )
            assert fig is not None

            # 测试 ROC 曲线
            fig = plot_roc_curve(y_true, y_proba, path=Path(tmpdir) / "roc.png")
            assert fig is not None

            # 测试混淆矩阵
            fig = plot_confusion_matrix(y_true, y_pred, path=Path(tmpdir) / "cm.png")
            assert fig is not None

        logger.info("✓ 可视化工具测试通过")
        return True
    except Exception as e:
        logger.error("✗ 可视化工具测试失败: %s", e)
        return False


def test_mlflow() -> bool:
    """测试 MLflow 实验记录."""
    logger.info("[5/5] 测试 MLflow 实验追踪...")
    try:
        from src.experiment import (
            log_metrics_from_dict,
            log_params_from_dict,
            setup_mlflow,
            start_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_uri = f"file://{tmpdir}/mlruns"
            experiment_name = "verify_setup"

            exp_id = setup_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
            assert exp_id is not None

            with start_run(run_name="test_run"):
                log_params_from_dict({"param_a": 1, "param_b": "hello"})
                log_metrics_from_dict({"metric_a": 0.95, "metric_b": 0.88})

        logger.info("✓ MLflow 测试通过")
        return True
    except Exception as e:
        logger.error("✗ MLflow 测试失败: %s", e)
        return False


def main() -> int:
    """主入口：运行所有验证测试."""
    logger.info("=" * 50)
    logger.info("开始基础设施验证")
    logger.info("=" * 50)

    results = [
        test_imports(),
        test_data_utils(),
        test_model_utils(),
        test_viz_utils(),
        test_mlflow(),
    ]

    logger.info("=" * 50)
    passed = sum(results)
    total = len(results)
    if all(results):
        logger.info("✓ 全部 %d/%d 项测试通过，基础设施验证成功！", passed, total)
        return 0
    else:
        logger.error("✗ 部分测试失败: %d/%d 通过", passed, total)
        return 1


if __name__ == "__main__":
    sys.exit(main())
