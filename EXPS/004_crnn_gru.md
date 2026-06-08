# Experiment 004: CRNN GRU log-mel baseline

Goal: test whether explicit temporal modeling improves over a CNN baseline.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/crnn_gru.yaml'
```

Final status: completed.

Run directory on JH:

```text
/home/margasanov/temki/diplom/reports/runs/20260607_085956_crnn_gru_logmel
```

Model info:

| item | value |
|---|---:|
| parameters | 2063300 |
| device | cuda |

Logged artifacts:

- `config.yaml`
- `model_info.json`
- `metrics.csv`
- `checkpoints/best.pt`
- `test_metrics.json`
- confusion matrix figure
- training curves figure

Test metrics:

| metric | value |
|---|---:|
| accuracy | 0.3901 |
| macro F1 | 0.2998 |
| weighted F1 | 0.4080 |
| balanced accuracy | 0.3062 |
| test loss | 2.7724 |

Per-class recall:

| class | recall |
|---|---:|
| normal | 0.4776 |
| crackle | 0.3850 |
| wheeze | 0.2500 |
| both | 0.1122 |

Interpretation:

The CRNN is more compact than ResNet18 and recovers more `normal` examples, but it does not improve the difficult minority classes enough. It is behind ResNet18 on macro F1 and balanced accuracy.

Conclusion: temporal GRU modeling is useful but not sufficient here; the result is a middle baseline between MFCC and ResNet18.
