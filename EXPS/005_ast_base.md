# Experiment 005: AST-Base fine-tuning

Goal: fine-tune AST-Base under the 100M parameter limit as the main transformer baseline.

Prerequisite:

Sync pretrained assets to:

```text
/opt/gen-content/margasanov/hf_models/ast-finetuned-audioset-10-10-0.4593
```

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base.yaml'
```

Final status: completed.

Run directory on JH:

```text
/home/margasanov/temki/diplom/reports/runs/20260607_091747_ast_base
```

Model info:

| item | value |
|---|---:|
| parameters | 86191876 |
| device | cuda |

Logged artifacts:

- `config.yaml`
- `model_info.json`
- `metrics.csv`
- `checkpoints/best.pt`
- `test_metrics.json`
- `figures/confusion_matrix.png`
- `figures/training_curves.png`

Test metrics:

| metric | value |
|---|---:|
| accuracy | 0.5776 |
| macro F1 | 0.4867 |
| weighted F1 | 0.5817 |
| balanced accuracy | 0.4928 |
| test loss | 1.8926 |

Per-class recall:

| class | recall |
|---|---:|
| normal | 0.6619 |
| crackle | 0.5241 |
| wheeze | 0.5608 |
| both | 0.2245 |

Comparison with baselines:

| model | macro F1 | balanced accuracy | weighted F1 |
|---|---:|---:|---:|
| MFCC + logistic regression | 0.2878 | 0.2882 | 0.4325 |
| ResNet18 log-mel | 0.3044 | 0.3396 | 0.3865 |
| CRNN GRU log-mel | 0.2998 | 0.3062 | 0.4080 |
| AST-Base | 0.4867 | 0.4928 | 0.5817 |

Interpretation:

AST-Base is the clear best model across all reported aggregate metrics. It also improves every per-class recall compared with the baselines, including the rare `both` class. The `both` class remains the weakest class, but AST raises its recall from `0.1122` in the best non-transformer runs to `0.2245`.

Conclusion: AST-Base is the final selected architecture for the current experimental setup.
