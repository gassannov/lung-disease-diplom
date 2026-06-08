# Experiment 007: AST-Base without imbalance handling

Goal: measure the effect of training on the original imbalanced train split without weighted loss, weighted sampler, focal loss, or augmentation.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_unbalanced.yaml'
```

Final status: completed with manual early stop after epoch 29 because validation macro F1 did not improve after epoch 22.

Run directory on JH:

```text
/home/margasanov/temki/diplom/reports/runs/20260608_131349_ast_base_unbalanced
```

Configuration:

| item | value |
|---|---|
| loss | cross entropy |
| weighted loss | false |
| weighted sampler | false |
| augmentation | false |
| seed | 42 |

Expected comparison:

This run is the control point for the imbalance study. It should show how strongly the model shifts toward frequent classes when the train distribution remains unchanged and no balancing method is applied.

Test metrics:

| metric | value |
|---|---:|
| accuracy | 0.5000 |
| macro F1 | 0.4657 |
| weighted F1 | 0.5181 |
| balanced accuracy | 0.5029 |

Per-class recall:

| class | recall |
|---|---:|
| normal | 0.4597 |
| crackle | 0.7754 |
| wheeze | 0.3581 |
| both | 0.4184 |

Interpretation:

Without explicit balancing, AST-Base does not simply collapse to `normal`; the best checkpoint becomes sensitive to `crackle` and `both`, but loses much of the majority-class recall and weighted F1. This confirms that the imbalance problem cannot be judged only by accuracy or one minority-class recall value.

Comparison with weighted AST:

| run | macro F1 | balanced accuracy | weighted F1 | recall both |
|---|---:|---:|---:|---:|
| unbalanced AST | 0.4657 | 0.5029 | 0.5181 | 0.4184 |
| weighted AST | 0.4867 | 0.4928 | 0.5817 | 0.2245 |

Conclusion: the weighted setup remains preferable as the main model because it improves macro F1 and weighted F1, but the unbalanced control shows that minority recall can increase at the cost of unstable class trade-offs.
