# Experiment 008: AST-Base with minority-class augmentation

Goal: evaluate class balancing with waveform and spectrogram augmentation for minority classes.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_augmented.yaml'
```

Final status: prepared for run.

Augmentation policy:

| level | method | value |
|---|---|---|
| waveform | random gain | 0.85--1.15 |
| waveform | circular time shift | up to 250 ms |
| waveform | Gaussian noise | 0.003 of segment std |
| log-mel | frequency mask | up to 12 mel bins |
| log-mel | time mask | up to 48 frames |

Class policy:

Augmentation is applied only to train examples with labels `crackle`, `wheeze`, and `both`. Validation and test examples are never augmented.

Expected comparison:

This run should be compared with `ast_base_unbalanced`, `ast_base`, and `ast_base_focal` by macro F1, balanced accuracy, and recall/F1 for `both`.

