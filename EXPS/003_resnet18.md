# Experiment 003: ResNet18 log-mel baseline

Goal: evaluate local convolutional patterns on 128-bin log-mel spectrograms.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/resnet18.yaml'
```

Final status: completed.

Run directory on JH:

```text
/home/margasanov/temki/diplom/reports/runs/20260607_084148_resnet18_logmel
```

Model info:

| item | value |
|---|---:|
| parameters | 11172292 |
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
| accuracy | 0.3649 |
| macro F1 | 0.3044 |
| weighted F1 | 0.3865 |
| balanced accuracy | 0.3396 |
| test loss | 2.9441 |

Per-class recall:

| class | recall |
|---|---:|
| normal | 0.3596 |
| crackle | 0.4813 |
| wheeze | 0.4054 |
| both | 0.1122 |

Interpretation:

ResNet18 sacrifices majority-class `normal` recall, but improves recognition of `crackle` and `wheeze`. It has the best balanced accuracy among the completed non-transformer models, which makes it the strongest CNN-family baseline.

Conclusion: best non-transformer baseline by balanced accuracy and macro F1, but still weak on the rare `both` class.
