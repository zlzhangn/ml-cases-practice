# AGENTS.md

## Project Overview

This is `ml-cases-practice`: a hands-on machine learning learning project with 7 real-world case studies covering classification, regression, time series, anomaly detection, NLP, causal inference, and quantitative backtesting.

## Key Documents

All project-level decisions and specifications live in `specs/`:

| File | Purpose |
|------|---------|
| [`specs/mission.md`](specs/mission.md) | Project mission, goals, target audience, scope boundaries, success criteria |
| [`specs/tech-stack.md`](specs/tech-stack.md) | Technology stack, dependencies, directory structure, naming conventions, coding standards |
| [`specs/roadmap.md`](specs/roadmap.md) | 7-week execution roadmap, phase divisions, acceptance criteria, risk mitigation |

## Quick Reference

- **Language**: Python 3.11+, all documentation and comments in Chinese
- **Package Manager**: `uv` (Rust-based, replaces pip/conda)
- **Dataset Access**: Hugging Face mirror (hf-mirror.com) allowed for dataset downloads
- **Core Libraries**: pandas, numpy, scikit-learn, LightGBM, XGBoost, SHAP, MLflow, Optuna
- **Directory Pattern**: `0X-case-name/` (e.g., `01-customer-churn`)
- **Case Structure**: Each case has `config.yaml`, `notebooks/`, `src/`, `outputs/`, `mlruns/`
- **Development Principle**: Fast-pass first — complete the full pipeline, optimize later

## Agent Guidelines

1. **Read specs first** before making any project-level decisions
2. **Follow existing conventions** — directory structure, naming, code style are all defined in `tech-stack.md`
3. **Use Chinese** for all documentation, comments, and user-facing text
4. **Keep it simple** — this is a learning project, not production infrastructure
5. **Verify dataset availability** on HuggingFace before committing to a data source
