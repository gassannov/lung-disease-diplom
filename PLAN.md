# План практической реализации диплома

Цель практической части: решить задачу классификации дыхательных аудиозаписей на сыром датасете около `5GB`, сравнить четыре семейства подходов и выбрать лучшую базовую архитектуру для domain-specific обучения при ограничении `100M` параметров.

## Итоговый артефакт

Практическая часть должна дать:

- подготовленный pipeline обработки сырого аудиодатасета;
- baseline на ручных признаках;
- CNN/ResNet baseline на спектрограммах;
- CRNN baseline: CNN + RNN/LSTM/GRU;
- transformer baseline;
- выбранную базовую архитектуру `AST-Base`;
- domain-specific fine-tuning выбранной архитектуры;
- таблицу метрик и анализ ошибок;
- выводы для диплома о качестве и ограничениях метода.

## Основная постановка

Задача: 4-class respiratory sound classification.

Классы:

- `normal`;
- `crackle`;
- `wheeze`;
- `both`.

Основной датасет: сырой набор дыхательных аудиозаписей объемом около `5GB`. Если в датасете есть patient-level metadata, split должен быть patient-independent. Если такой metadata нет, split фиксируется по файлам с явным описанием ограничения.

Основные метрики:

- macro F1;
- balanced accuracy;
- per-class recall;
- weighted F1;
- confusion matrix.

Ограничение модели: не более `100M` параметров. Это исключает крупные foundation-модели как основной практический вариант и оставляет реалистичными ResNet, CRNN и `AST-Base`. Выбранная модель `AST-Base` имеет порядок `86M` параметров, поэтому проходит лимит и остается достаточно сильной базой для fine-tuning.

## Этапы

### Этап 1. Подготовка проекта

Сделать структуру:

```text
data/
  raw/
  processed/
configs/
src/
  data/
  features/
  models/
  train.py
  evaluate.py
notebooks/
reports/
```

Что должно быть в коде:

- чтение WAV;
- чтение аннотаций;
- сбор index table;
- train/validation/test split;
- сохранение metadata CSV.

### Этап 2. Подготовка данных

Pipeline:

1. Разместить сырой датасет в `data/raw/`.
2. Проверить общий размер датасета и формат файлов.
3. Распарсить annotations.
4. Нарезать или загрузить размеченные аудиофрагменты.
5. Привести audio к 16 kHz mono.
6. Сформировать fixed-length segments через padding/cropping.
7. Сохранить index table:

```text
sample_id, file, start, end, class, split, duration, source, patient_id
```

### Этап 3. Baseline 1: ручные признаки

Признаки:

- MFCC 40;
- delta и delta-delta;
- mean/std/min/max по времени;
- spectral centroid/bandwidth/rolloff;
- RMS и ZCR.

Модели:

- logistic regression;
- SVM или Random Forest.

Цель: получить нижнюю границу качества и показать, что ручных признаков недостаточно.

### Этап 4. Baseline 2: CNN/ResNet на спектрограммах

Вход:

- log-mel spectrogram;
- 128 mel bins;
- 25 ms window;
- 10 ms hop;
- 8 seconds.

Модель:

- ResNet18 или ResNet50;
- classifier head на 4 класса;
- weighted cross entropy.

Аугментации:

- time/frequency masking;
- additive noise;
- random crop/cyclic shift.

Цель: проверить, насколько хорошо локальные сверточные признаки работают на log-mel представлении.

### Этап 5. Baseline 3: CRNN

Вход:

- log-mel spectrogram;
- те же параметры preprocessing, что у CNN baseline.

Модель:

- сверточный encoder;
- BiLSTM или GRU;
- temporal pooling;
- classifier head.

Цель: проверить, дает ли явное моделирование временной структуры улучшение относительно CNN/ResNet.

### Этап 6. Baseline 4: Transformer approach

- AST-Base pretrained on AudioSet;
- заменить classification head на 4 класса;
- fine-tune 50 epochs.

Параметры из литературы:

```text
sample_rate = 16000
duration = 8s
mel_bins = 128
window = 25ms
hop = 10ms
batch_size = 8
optimizer = Adam/AdamW
lr = 5e-5 или 5e-4 для adapter-style частей
epochs = 50
loss = weighted cross entropy
```

Цель: проверить transformer-подход как наиболее сильный кандидат при ограничении `100M` параметров.

### Этап 7. Выбор лучшего подхода

Сравнить четыре семейства:

| Подход | Сильная сторона | Слабая сторона | Роль в дипломе |
|---|---|---|---|
| Ручные признаки + ML | простота и интерпретируемость | слабее извлекает сложные паттерны | нижняя граница |
| CNN/ResNet | сильный baseline для спектрограмм | хуже моделирует длинный контекст | сверточная база |
| CRNN | учитывает временную структуру | сложнее и медленнее CNN | временной baseline |
| Transformer/AST | pretrained audio representation и global context | дороже по памяти | выбранный основной подход |

Выбранный подход: `AST-Base`.

Причины выбора:

- укладывается в ограничение `100M` параметров;
- имеет порядок `86M` параметров;
- использует pretraining на большом аудиодатасете;
- работает напрямую с log-mel spectrogram;
- лучше подходит для domain-specific fine-tuning, чем обучение CNN/CRNN с нуля на `5GB` raw data;
- дает понятную архитектуру для сравнения с более простыми baseline.

### Этап 8. Domain-specific обучение выбранной архитектуры

Реализовать fine-tuning `AST-Base`:

- AST-Base pretrained on AudioSet;
- classifier head под классы датасета;
- weighted sampler или class-balanced sampler;
- weighted cross entropy или focal loss как ablation;
- same preprocessing, что у baseline, чтобы сравнение было честным.

Стартовый конфиг:

```text
sample_rate = 16000
duration = 8s
mel_bins = 128
window = 25ms
hop = 10ms
batch_size = 8
epochs = 50
optimizer = AdamW
lr = 5e-5
loss = weighted cross entropy
max_params = 100M
```

### Этап 9. Анализ результатов

Сделать таблицу:

| Метод | Macro F1 | Balanced Acc | Per-class Recall | Params | Комментарий |
|---|---:|---:|---:|---:|---|
| MFCC + SVM/LogReg | | | | | ручные признаки |
| ResNet18/50 | | | | | CNN baseline |
| CRNN | | | | | temporal baseline |
| AST-Base | | | | <=100M | выбранная архитектура |
| AST-Base fine-tuned | | | | <=100M | domain-specific обучение |

Сделать графики:

- confusion matrix;
- per-class recall;
- training/validation loss;
- macro F1 по seed;
- примеры спектрограмм с правильными и ошибочными предсказаниями.

### Этап 10. Выводы

Ответить в дипломе:

- какие методы лучше подходят для маленького шумного аудиодатасета;
- насколько transfer learning помогает;
- почему per-class recall остается сложной метрикой;
- какие классы путаются чаще всего;
- какие ограничения есть у размера данных, разметки и воспроизводимости;
- что нужно для дальнейшего улучшения экспериментального pipeline.

## Риски

- Малый и несбалансированный датасет: использовать weighted loss, balanced sampler, macro metrics.
- Data leakage: использовать official patient-independent split.
- Плохая воспроизводимость сравнения: фиксировать seed, версии библиотек, параметры preprocessing.
- Слишком долгий fine-tuning: сначала заморозить backbone/head-only, затем full/partial fine-tuning.
- Ошибочное позиционирование задачи: писать "классификация аудиосегментов по разметке датасета", а не "автоматическая постановка диагноза".

## Минимальный успешный результат

Если времени мало, практическая часть считается достаточной при наличии:

1. Preprocessing сырого датасета.
2. MFCC baseline.
3. CNN/ResNet или CRNN baseline.
4. AST-Base fine-tuning.
5. Метрики macro F1, balanced accuracy, per-class recall.
6. Confusion matrix.
7. Анализ ошибок и ограничений.

## Расширенный результат

Если времени достаточно:

- добавить ResNet baseline;
- добавить CRNN baseline;
- запустить 3-5 seed;
- сделать ablation: ResNet18 vs ResNet50;
- сделать ablation: LSTM vs GRU в CRNN;
- сделать ablation: AST frozen backbone vs full fine-tuning;
- сделать binary normal/abnormal experiment;
- проверить перенос на отдельный subset, если в датасете есть разные источники.
