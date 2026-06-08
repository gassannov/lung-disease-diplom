# Experiment 002: MFCC + logistic regression

Goal: establish a lower-bound baseline with manual MFCC and spectral statistics.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ml_mfcc_logreg.yaml'
```

Final status: completed.

Run directory on JH:

```text
/home/margasanov/temki/diplom/reports/runs/20260607_083724_ml_mfcc_logreg
```

Logged artifacts:

- `model.joblib`
- `val_metrics.json`
- `test_metrics.json`
- confusion matrix figure

Test metrics:

| metric | value |
|---|---:|
| accuracy | 0.4194 |
| macro F1 | 0.2878 |
| weighted F1 | 0.4325 |
| balanced accuracy | 0.2882 |

Per-class recall:

| class | recall |
|---|---:|
| normal | 0.5796 |
| crackle | 0.2888 |
| wheeze | 0.2027 |
| both | 0.0816 |

Interpretation:

This baseline has the best ordinary accuracy among the non-transformer baselines, but this is mostly because it predicts the majority class better than the rare classes. Its balanced accuracy and macro F1 are weak, and the `both` class is almost not recognized.

Conclusion: useful as a simple lower-bound baseline, but not strong enough for the final model choice.
