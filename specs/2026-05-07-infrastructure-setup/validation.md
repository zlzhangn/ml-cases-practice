# Validation: 项目基础设施搭建

## 验证方法

### 1. 环境安装验证

```bash
# 在干净环境中执行
uv sync
```

**成功标准**：命令执行成功，所有依赖安装完成，无报错。

### 2. 模块导入验证

```bash
python -c "from src import data_utils, model_utils, viz_utils, experiment"
```

**成功标准**：无 ImportError，所有模块能正常导入。

### 3. 示例脚本验证

运行提供的示例脚本（如 `examples/demo_pipeline.py`）：

```bash
python examples/demo_pipeline.py
```

**成功标准**：
- 脚本正常执行，无异常抛出
- 输出包含预期的日志信息（数据加载成功、模型训练完成、评估指标值）
- 在 `mlruns/` 目录下生成 MLflow 实验记录
- 在 `outputs/` 目录下生成示例图表

### 4. 代码质量验证

```bash
ruff check src/
ruff format --check src/
```

**成功标准**：无 lint 错误，格式符合规范。

### 5. 目录结构验证

检查项目根目录是否符合以下结构：

```
├── configs/
│   └── base_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── viz_utils.py
│   └── experiment.py
├── pyproject.toml
├── uv.lock
└── README.md
```

**成功标准**：所有文件和目录存在，且命名符合规范。

## 合并条件

以下**全部满足**时，方可合并到 `main` 分支：

1. [ ] `uv sync` 在新环境中能成功执行
2. [ ] `python -c "from src import data_utils, model_utils, viz_utils, experiment"` 通过
3. [ ] 示例脚本成功运行，输出预期结果
4. [ ] `mlruns/` 目录下有 MLflow 实验记录生成
5. [ ] `ruff check src/` 无错误
6. [ ] `ruff format --check src/` 无格式问题
7. [ ] 代码审查通过（如适用）

## 验证责任人

- **开发者**：执行上述验证步骤，确保所有检查项通过
- **审阅者**（如适用）：在独立环境中复现验证步骤，确认可复现性

## 验证失败处理

若验证未通过：
1. 记录失败的具体步骤和错误信息
2. 在功能分支上修复问题
3. 重新执行全部验证步骤
4. 禁止部分验证通过后合并
