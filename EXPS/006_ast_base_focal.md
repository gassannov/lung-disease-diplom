# Experiment 006: AST-Base focal-loss imbalance handling

Goal: compare weighted cross entropy with focal loss for class imbalance.

Prerequisite:

Use the same AST local checkpoint as experiment 005.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_focal.yaml'
```

Final status: prepared for run.

Reference AST baseline:

| run | macro F1 | balanced accuracy | weighted F1 |
|---|---:|---:|---:|
| `20260607_091747_ast_base` | 0.4867 | 0.4928 | 0.5817 |

Expected artifacts after the run:

- `metrics.csv`
- `test_metrics.json`
- `figures/confusion_matrix.png`
- `figures/training_curves.png`
- `checkpoints/best.pt`

Conclusion: this run checks whether focal loss improves the rare `both` class without reducing overall macro F1.
