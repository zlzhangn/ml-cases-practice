# Plan: 项目基础设施搭建

## 概述

本计划对应路线图 Phase 0：搭建项目骨架，为后续 7 个案例提供统一的代码基座和开发环境。

## 任务组

### 1. 初始化项目配置

1.1 创建 `pyproject.toml`，定义项目元数据、依赖列表和工具配置（ruff）
1.2 创建 `uv.lock` 依赖锁定文件（通过 `uv lock` 生成）
1.3 验证 `uv sync` 能一键安装所有依赖

### 2. 创建公共工具库（`src/`）

2.1 创建 `src/__init__.py`，使 `src` 成为可导入包
2.2 创建 `src/data_utils.py`，封装数据加载、缓存、预处理等公共函数
2.3 创建 `src/model_utils.py`，封装模型训练、评估、保存的统一接口
2.4 创建 `src/viz_utils.py`，封装常用可视化图表模板
2.5 创建 `src/experiment.py`，封装 MLflow 实验启动、参数记录、指标追踪

### 3. 创建全局配置模板

3.1 创建 `configs/base_config.yaml`，定义各案例通用的默认配置项
3.2 定义配置结构：数据路径、模型参数、评估阈值、随机种子等

### 4. 验证与集成测试

4.1 编写一个示例脚本，验证能成功导入 `src` 下所有模块
4.2 验证 MLflow 能正常启动并记录实验
4.3 运行示例脚本，确认数据加载、模型训练、可视化、实验记录全流程正常

### 5. 文档与规范

5.1 更新 `README.md`，说明项目结构、环境搭建步骤、运行方式
5.2 确认 `.gitignore` 已正确配置（数据、模型、缓存等忽略规则）

## 退出条件

- `uv sync` 成功执行且无报错
- `python -c "from src import data_utils, model_utils, viz_utils, experiment"` 通过
- 示例脚本运行成功，输出预期结果
- 所有代码通过 `ruff check` 和 `ruff format`
