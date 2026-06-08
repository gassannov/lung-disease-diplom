#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/opt/gen-content/margasanov/envs/pizding-kartocheck"

conda run -p "${ENV_PATH}" python -m src.data.index --config configs/data/default.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/ml_mfcc_logreg.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/resnet18.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/crnn_gru.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/ast_base_unbalanced.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/ast_base.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/ast_base_focal.yaml
conda run -p "${ENV_PATH}" python -m src.train --config configs/experiments/ast_base_augmented.yaml
conda run -p "${ENV_PATH}" python -m src.report --runs-dir reports/runs --output reports/summary.csv
