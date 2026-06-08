# Experiment 001: prepare data index

Goal: build a reproducible segment-level index from raw ICBHI respiratory audio and annotation files.

Command on JH:

```bash
mutagen project flush
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.data.index --config configs/data/default.yaml'
```

Final status: completed.

Dataset location on JH:

```text
/opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database
```

Output artifact:

```text
/home/margasanov/temki/diplom/data/processed/index.csv
```

Index summary:

| split | segments |
|---|---:|
| train | 4915 |
| val | 991 |
| test | 992 |
| total | 6898 |

Train class distribution:

| class | segments |
|---|---:|
| normal | 2597 |
| crackle | 1323 |
| wheeze | 655 |
| both | 340 |

Test class distribution:

| class | segments |
|---|---:|
| normal | 559 |
| crackle | 187 |
| wheeze | 148 |
| both | 98 |

Training data size:

- Raw train segment duration: `13199.78` seconds, or `3.67` hours.
- Model-facing fixed train duration: `4915 * 8s = 39320` seconds, or `10.92` hours.

One training sample:

```text
sample_id: 102_1b1_Ar_sc_Meditron_0000
file: /opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database/...
start: 0.264
end: 1.736
class: normal
label: 0
split: train
duration: 1.472
patient_id: 102
```

Model input representation:

| model family | input tensor |
|---|---|
| waveform | `[128000]` |
| ResNet/CRNN log-mel | `[1, 128, 801]` |
| AST | `[1024, 128]` |

Conclusion: the prepared dataset is usable for all four experiment families. The main limitation is strong class imbalance, especially for `both`, which has only `340` train segments and `98` test segments.
