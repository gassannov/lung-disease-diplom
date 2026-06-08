# Ресерч по теме диплома

Тема: "Прогнозирование респираторной болезни на основе анализа звукозаписи дыхания".

Дата ресерча: 2026-06-06.

## Короткий итог

Для диплома лучше формулировать задачу как исследование автоматического анализа дыхательных аудиозаписей: сначала распознавание патологических дыхательных событий, затем оценка пригодности такого подхода для прогнозирования/скрининга респираторного заболевания. Это снимает вопрос "нужно ли придумывать свое решение": для дипломной работы достаточно провести обзор области, выбрать современный класс методов, реализовать воспроизводимый эксперимент, сравнить с базовыми подходами и проанализировать ограничения.

Основной рекомендуемый датасет: ICBHI 2017 Respiratory Sound Database. Он небольшой, публичный, стандартный для этой области и содержит 920 аудиозаписей от 126 пациентов, 5.5 часа звука и 6898 размеченных дыхательных циклов. В исходном виде размер зависит от частот дискретизации файлов, но после приведения к 16 kHz mono 16-bit весь звук занимает примерно 0.63 GB, что укладывается в ограничение около 1 GB для рабочих сырых/нормализованных данных.

Рекомендуемая архитектура для практической части по состоянию на 2026 год: `SAM-Optimized AST`, то есть Audio Spectrogram Transformer с Sharpness-Aware Minimization и взвешенной выборкой для борьбы с дисбалансом. Это самая сильная найденная single-model постановка на ICBHI 2017: заявлены `Score=68.10%` и `Sensitivity=68.31%`. Для сравнения стоит включить `ADD-RSC` из Interspeech 2025 как сильный denoising baseline (`Score=65.53%`) и `Meta-Ensemble Learning` из EMBC 2026 как самый свежий ансамблевый подход (`Score=66.49%`), но выбранной "полноценной архитектурой" для диплома лучше сделать именно `SAM-Optimized AST`, потому что это одна воспроизводимая модель, а не ансамбль моделей на разных split.

Ограничение 0.5B параметров выполняется: AST-Base обычно находится примерно в диапазоне 86-90M параметров. Даже если брать крупные health-acoustic foundation encoders вроде HeAR как источник embeddings, практическая дипломная модель может быть обучаемым head/stacking-классификатором поверх frozen embeddings, а не full fine-tuning огромной модели.

## Актуализация на 2026 год

По состоянию на 2026-06-06 свежий ландшафт выглядит так:

- `SAM-Optimized AST` - наиболее сильный найденный single-model результат для ICBHI event classification: `Score=68.10%`, `Sensitivity=68.31%`. Это не просто еще один Transformer, а geometry-aware optimization: SAM оптимизирует не только значение loss, но и устойчивость минимума, что важно на малом шумном медицинском датасете.
- `Meta-Ensemble Learning with Diverse Data Splits` - EMBC 2026 accepted работа, заявляет новый SOTA `Score=66.49%` на ICBHI и улучшенную out-of-distribution проверку. Это актуально для раздела "новейшие методы 2026", но как практическое ядро диплома сложнее: нужно обучать несколько base models на разных split и meta-model.
- `DPAST-LS` - 2026 Biomedical Signal Processing and Control, AST + transfer learning + global pruning + self-distillation. Самая актуальная lightweight/compression ветка: около 90% сокращения параметров, сильные SPRSound результаты, но на ICBHI 4-class `Score=58.37%`, поэтому не лучший выбор как основная модель качества.
- `Hybrid LSTM-KAN` - 2026 Computer Methods and Programs in Biomedicine Update. Актуальная disease-level ветка: LSTM encoder + Kolmogorov-Arnold Network, focal loss, SMOTE, targeted augmentation; заявлены `Accuracy=94.6%` и `Macro F1=0.703` на ICBHI disease classes. Это другая постановка, не official ICBHI crackle/wheeze event score.
- `PulmoVec / HeAR` - 2026 направление foundation health acoustics: HeAR embeddings + stacking для SPRSound pediatric tasks. Очень актуально для обзора будущих работ и pediatric screening, но основной датасет другой.
- Клинические 2026 работы усиливают акцент на real-world validation: ICU normal/abnormal respiratory sounds, multi-channel MIL, asthma diagnosis, новые acoustic repository protocols.

Вывод: для диплома на 2026 год лучше писать, что современный фронтир состоит из трех направлений: `AST + robust optimization`, `foundation embeddings + stacking/ensembles`, `lightweight distilled AST`. Для реализации выбрать `SAM-Optimized AST`, а `Meta-Ensemble`, `DPAST-LS` и `HeAR/PulmoVec` описать как актуальные 2026 расширения.

## Что именно предсказывать

В области есть две близкие, но разные постановки.

1. Распознавание дыхательных событий: классификация дыхательного цикла на `normal`, `crackle`, `wheeze`, `both`. Это основная задача ICBHI. Она хорошо подходит для честного бенчмарка, потому что есть официальный split и устоявшиеся метрики.

2. Распознавание патологического состояния: `healthy/unhealthy`, COPD, pneumonia, URTI, bronchiectasis и т.д. В ICBHI такие метки есть на уровне пациента/записи, но они крайне несбалансированы: например, COPD доминирует, asthma и LRTI представлены единичными случаями. Поэтому для диплома лучше не обещать полноценную клиническую диагностику всех болезней, а писать: "исследуется качество автоматического выявления патологических дыхательных паттернов как признаков респираторного заболевания".

Практически надежная формулировка цели:

> Цель работы - исследовать качество методов глубокого обучения при классификации дыхательных аудиозаписей на нормальные и патологические/адвентивные дыхательные паттерны, а также оценить применимость выбранной современной архитектуры для скрининга респираторных заболеваний.

## Актуальность

Респираторные заболевания остаются большой медицинской проблемой. WHO отмечает, что хронические респираторные заболевания входят в число распространенных неинфекционных заболеваний; COPD является одной из ведущих причин смерти в мире. Аускультация легких остается дешевым и неинвазивным способом первичной оценки состояния дыхательной системы, но ручная интерпретация зависит от опыта врача, качества записи и шума. Автоматический анализ аудиосигналов может помогать в первичном скрининге, удаленном мониторинге и стандартизации оценки.

Для диплома важно честно указать: модель не заменяет врача и не ставит диагноз сама по себе. Она оценивает акустические признаки, связанные с патологией: wheeze, crackle, rhonchi, stridor, abnormal/normal.

## Классы архитектур

Ниже не больше пяти классов, от старых к современным.

### 1. Ручные признаки + классические ML-модели

Идея: из аудио извлекаются признаки, затем обучается SVM, Random Forest, kNN, HMM/GMM, логистическая регрессия или бустинг.

Типичные признаки:

- time-domain: среднее, дисперсия, RMS, zero crossing rate, длительности фаз;
- frequency-domain: spectral centroid, bandwidth, roll-off, power spectral density;
- time-frequency: STFT, MFCC, log-mel spectrogram, wavelet transform, CQT;
- нелинейные признаки: entropy, fractal-like descriptors.

Плюсы: простота, интерпретируемость, быстрые эксперименты, хороший baseline.

Минусы: признаки зависят от ручного дизайна, плохо переносятся между устройствами, обычно уступают глубоким моделям на сложных шумных данных.

Для диплома: обязательно реализовать один простой baseline, например `log-mel/MFCC + logistic regression/SVM`, чтобы показать, зачем нужна глубокая модель.

### 2. CNN/ResNet на спектрограммах

Идея: аудио переводится в двумерное представление, чаще всего log-mel spectrogram, и подается как изображение в CNN, ResNet, VGG, MobileNet, EfficientNet или специализированные lung-sound CNN.

Плюсы:

- хорошо извлекают локальные time-frequency паттерны;
- crackle и wheeze визуально проявляются как локальные структуры на спектрограмме;
- можно использовать ImageNet-pretraining;
- компактные CNN подходят для edge/mobile.

Минусы:

- локальная свертка хуже моделирует длинные зависимости дыхательного цикла;
- ImageNet-pretraining не идеально совпадает с аудиодоменом;
- при маленьком датасете легко переобучаются.

Сильные представители в области: LungRN+NL, RespireNet, ResNet/ResNeSt с domain transfer, lightweight CNN на scalogram/log-mel.

### 3. CRNN: CNN + RNN/LSTM/GRU

Идея: CNN извлекает локальные признаки по спектрограмме или waveform, а RNN/LSTM/GRU моделирует временную динамику дыхательного цикла.

Плюсы:

- естественная работа с последовательностями;
- полезно для breath phase detection, event detection, inhalation/exhalation;
- применялось на HF_Lung для CAS/DAS и фаз дыхания.

Минусы:

- сложнее обучать и настраивать;
- RNN хуже масштабируются, чем attention/transformer;
- для ICBHI 4-class сейчас уже не самый сильный класс.

Для диплома CRNN можно описать в обзоре, но не обязательно реализовывать, если цель - сравнить baseline, CNN и современный Transformer.

### 4. Audio Transformers и pretrained audio encoders

Идея: спектрограмма режется на патчи, каждый патч превращается в token, дальше используется self-attention. Основные представители: AST, SSAST, HTS-AT, AudioMAE, BEATs, M2D, CLAP/LAION-CLAP, OPERA.

Почему это важно:

- self-attention видит глобальные зависимости;
- pretrained модели компенсируют малый размер медицинского датасета;
- transfer learning сейчас практически обязателен для ICBHI-подобных задач.

Ключевые модели:

- AST: Audio Spectrogram Transformer, чисто attention-based audio classifier, сильная базовая архитектура для log-mel spectrogram.
- SSAST: self-supervised AST с masked spectrogram modeling.
- HTS-AT: hierarchical token-semantic audio transformer.
- BEATs: audio pretraining with acoustic tokenizers.
- OPERA: respiratory acoustic foundation benchmark/model family, специально для respiratory audio.
- HeAR: health-acoustic foundation encoder от Google, pretrained на 300+ million двухсекундных health-acoustic clips; в 2026 вокруг него появились pediatric respiratory pipelines вроде PulmoVec.

Минусы:

- чувствительны к длине входа и параметрам спектрограммы;
- полное fine-tuning может быть дорогим и склонным к переобучению;
- без хорошей аугментации и class balancing результат на ICBHI может быть слабым по sensitivity.

### 5. Современные respiratory-specific методы вокруг AST

Это самый релевантный класс для диплома. Он берет AST/pretrained encoder и добавляет доменно-специфичные механизмы:

- Patch-Mix CL: patch-level mixing + contrastive learning для борьбы с малым числом данных и дисбалансом;
- SG-SCL: supervised contrastive learning с учетом доменного сдвига стетоскопов;
- LungAdapter: parameter-efficient адаптеры в AST, всего около 2.48M trainable parameters при total около 90M;
- MVST: multi-view spectrogram transformer;
- BTS: bridging text and sound modalities с metadata;
- ADD-RSC: adaptive differential denoising поверх AST, сильный 2025 denoising baseline.
- SAM-Optimized AST: AST + Sharpness-Aware Minimization + weighted sampling; лучший найденный single-model ICBHI результат на 2026 год.
- Meta-Ensemble 2026: объединение base models, обученных на разных split/granularity settings, через meta-model; сильная ансамблевая ветка.

Итоговый выбор: `SAM-Optimized AST на AST-Base`.

## Выбранная архитектура

### Почему не просто самая новая по дате

В 2026 опубликованы компрессионные работы вроде DPAST-LS. Они интересны для lightweight deployment и pruning/self-distillation, но на ICBHI 4-class их заявленный `Score=58.37%`, что ниже сильных AST-методов 2023-2025. Для диплома лучше выбрать архитектуру, которая одновременно:

- свежая;
- укладывается в 0.5B параметров;
- имеет понятную статью и код;
- показывает лучший официальный результат на основном бенчмарке;
- математически удобно описывается.

Этим условиям лучше всего соответствует `SAM-Optimized AST`. `Meta-Ensemble` свежее как accepted 2026 методология, но это ансамбль, а не одна архитектура. `DPAST-LS` свежее как lightweight-компрессия, но слабее по ICBHI Score. `HeAR/PulmoVec` актуальнее для health foundation/pediatric respiratory screening, но его основной benchmark - SPRSound, а не ICBHI event classification.

### Вход модели

Сырой сигнал:

```text
x(t), t = 1..T
```

Предобработка:

- resampling к 16 kHz;
- нарезка по дыхательным циклам из annotation txt;
- fixed duration 8 секунд;
- короткие циклы дополняются cyclic padding, длинные обрезаются/семплируются;
- log-mel spectrogram: 128 mel bins, window 25 ms, hop 10 ms;
- z-score нормализация спектрограммы.

Итоговый вход:

```text
X in R^(F x T), где F=128, T примерно 100 * duration_seconds
```

### AST backbone

AST делит спектрограмму на патчи `16 x 16`, проецирует патчи в embedding-пространство, добавляет positional embedding и class token, затем обрабатывает последовательность Transformer encoder-блоками:

```text
Z_0 = [z_cls; PatchEmbed(X)] + E_pos
Z'_l = Z_{l-1} + MHSA(LN(Z_{l-1}))
Z_l = Z'_l + MLP(LN(Z'_l))
y_hat = Head(LN(Z_L))
```

Смысл для дыхания: crackles часто являются короткими дискретными высокочастотными событиями, wheezes - более длительными тональными/свистящими компонентами. Attention помогает связывать локальные частотно-временные признаки в пределах всего дыхательного цикла.

### SAM-Optimized AST

SAM-Optimized AST оставляет основную архитектуру AST, но меняет стратегию оптимизации. На малых медицинских аудиоданных обычный fine-tuning Transformer может сходиться к "острым" минимумам: train loss низкий, но модель плохо переносится на unseen patients и шумные записи. Sharpness-Aware Minimization решает это через оптимизацию параметров, устойчивых к локальным возмущениям.

Обычная оптимизация:

```text
min_theta L(theta)
```

SAM:

```text
min_theta max_{||epsilon|| <= rho} L(theta + epsilon)
```

Практически это означает два шага на batch:

1. найти локальное возмущение параметров в направлении градиента;
2. обновить исходные параметры так, чтобы loss был малым и в этой худшей локальной окрестности.

Для ICBHI это важно, потому что модель не должна запоминать устройство, шум или пациента. Она должна выделять устойчивые паттерны wheeze/crackle. Вторая часть метода - weighted sampling: редкие классы `wheeze` и `both` чаще попадают в batch, поэтому sensitivity растет сильнее, чем при обычной accuracy-driven оптимизации.

### ADD-RSC как сильный baseline

ADD-RSC добавляет не отдельный внешний denoiser, а обучаемое подавление шума внутри классификационной модели:

- Adaptive Frequency Filters (`AFF`);
- Differential Denoise Layer (`DDL`);
- bias-aware denoising loss.

Общая функция потерь в ADD-RSC:

```text
L = beta * L_bias_denoise + (1 - beta) * L_CE
```

где `L_CE` - cross entropy, `L_bias_denoise` - denoising-компонента со label smoothing, `beta` регулирует вклад denoising.

### Почему это хорошо для диплома

- Архитектура современная, но не чрезмерная.
- Можно описать математически: STFT/log-mel, patch embedding, self-attention, loss, метрики.
- Можно реализовать практически: AST доступен в PyTorch/Hugging Face, SAM реализуется как optimizer wrapper.
- Можно сделать сравнение: baseline ML, CNN/ResNet, AST fine-tuning, ADD-RSC, SAM-Optimized AST.
- Можно уложиться в разумный compute: batch size 8, 50 epochs, pretrained AST.

## Основные бенчмарки

### ICBHI 2017 Respiratory Sound Database

Главный бенчмарк для диплома.

Данные:

- 920 annotated audio samples;
- 126 subjects;
- 5.5 hours recordings;
- 6898 respiratory cycles;
- classes: normal, crackles, wheezes, both;
- heterogeneous devices and sample rates from 4 kHz to 44.1 kHz;
- duration 10-90 seconds per recording;
- official train/test split 60/40.

Классы циклов:

- normal: около 3642/3643;
- crackles: 1864;
- wheezes: 886;
- both: 506.

Метрики ICBHI:

```text
Sp = correctly recognized normal / total normal
Se = correctly recognized abnormal / total abnormal
Score = (Sp + Se) / 2
```

Для 4-class задачи abnormal объединяет `crackle`, `wheeze`, `both` при подсчете Se. Эта метрика важна, потому что accuracy на дисбалансном датасете может выглядеть хорошо, даже если модель плохо находит патологию.

Сильные результаты на official split:

| Метод | Год | Backbone | Sp | Se | Score |
|---|---:|---|---:|---:|---:|
| Patch-Mix CL | 2023 | AST | 81.66 | 43.07 | 62.37 |
| LungAdapter | 2024 | AST + adapters | 80.43 | 44.37 | 62.40 |
| CycleGuardian | 2025 | clustering + contrastive | 82.06 | 44.47 | 63.26 |
| M2D | 2024/2025 citation | masked audio pretraining | 81.51 | 45.08 | 63.29 |
| BTS | 2024 | text-sound metadata | 81.40 | 45.67 | 63.54 |
| ADD-RSC | 2025 | AST + adaptive denoising | 85.13 | 45.94 | 65.53 |
| Meta-Ensemble | 2026 | diverse split ensemble + meta-model | - | - | 66.49 |
| SAM-Optimized AST | 2025/2026 frontier | AST + SAM + weighted sampling | - | 68.31 | 68.10 |

Вывод: ICBHI остается сложным, потому что даже лучшие official split методы имеют sensitivity около 46%. Это хороший материал для анализа ограничений: дисбаланс, шум, heterogeneity устройств, small-data, patient-level leakage в кастомных split.

### SPRSound

Педиатрический датасет SJTU. Полезен как дополнительный датасет для внешней проверки, но не основной.

Особенности:

- первый open-access pediatric respiratory sound database;
- возраст: от 1 месяца до 18 лет;
- WAV + JSON annotations;
- record-level labels: Normal, CAS, DAS, CAS & DAS, Poor Quality;
- event-level labels: Normal, Rhonchi, Wheeze, Stridor, Coarse Crackle, Fine Crackle, Wheeze+Crackle;
- BioCAS 2022 initial release: 1949 train files + 734 test files;
- последующие версии 2023-2025 добавляют test sets и задачи compression/detection.

Для диплома: использовать как обзорный бенчмарк или optional external validation. Основная проблема - pediatric domain отличается от ICBHI по пациентам и условиям записи.

### HF_Lung_V1

Большой open-access датасет для lung sound analysis.

Особенности:

- 9765 audio files;
- каждый файл 15 секунд;
- labels для inhalation, exhalation;
- CAS labels: wheeze, stridor, rhonchi;
- DAS labels: crackles;
- подходит для event detection и фаз дыхания.

Минус для текущего диплома: суммарная длительность примерно 40.7 часа, в сыром виде это может быть больше желаемого 1 GB. Хорош как источник для обзора и future work, но не нужен для основной практики.

### RespiratoryDatabase@TR

Датасет на Mendeley для COPD severity analysis.

Особенности:

- 12-channel lung sounds per patient;
- labels COPD0-COPD4;
- short-term recordings at least 17 seconds;
- есть мультимодальный контекст: chest X-ray, pulmonary function test, questionnaire.

Для диплома: интересен, если хочется именно disease severity, но сложнее для чистого аудио-проекта и не является таким стандартным бенчмарком, как ICBHI.

### Coswara

Crowdsourced дыхание/кашель/голос для COVID-19.

Особенности:

- breathing, cough, sustained vowels, counting;
- metadata: age, gender, location, health status, comorbidities;
- 9 sound samples per participant;
- open repository.

Минус: это не стетоскопическая аускультация легких, а consumer-device respiratory audio. Для темы "звукозапись дыхания" можно упомянуть, но для архитектур lung sound classification лучше использовать ICBHI.

## Работа с данными

### Разметка и leakage

Критически важно делить данные по пациентам, а не случайно по циклам. Если циклы одного пациента попадают и в train, и в test, результат будет завышен. ICBHI official split уже patient-independent, поэтому его надо использовать для основного результата.

### Сегментация

В ICBHI есть txt annotations:

```text
cycle_start cycle_end crackle wheeze
```

По ним формируются классы:

- `0 0` -> normal;
- `1 0` -> crackle;
- `0 1` -> wheeze;
- `1 1` -> both.

Для disease-level эксперимента можно агрегировать предсказания циклов в предсказание записи/пациента:

```text
patient_score = mean(prob_abnormal over cycles)
patient_label = unhealthy if patient_score >= threshold
```

Но в дипломе лучше не смешивать это с основной ICBHI 4-class метрикой.

### Частота дискретизации

Рекомендуется 16 kHz. В LungAdapter, ADD-RSC, AST-SAM и HeAR-style pipelines используется 16 kHz или совместимый frontend. Это сохраняет достаточно информации для большинства respiratory events и не раздувает вход.

Осторожность: некоторые crackles могут иметь высокочастотные компоненты, поэтому не стоит агрессивно резать все выше 2 kHz без эксперимента. В современных работах 2025-2026 отдельно подчеркивается, что жесткая фильтрация может удалять патологические признаки; поэтому robust optimization, learned denoising и weighted sampling предпочтительнее грубого preprocessing.

### Длина входа

Рекомендуется 8 секунд на цикл/сегмент.

Причины:

- циклы ICBHI имеют разную длительность;
- AST требует fixed-size input;
- 8 секунд используется в сильных AST работах;
- короткие циклы можно расширять cyclic padding, чтобы не добавлять только тишину.

### Представление аудио

Основной вариант:

- log-mel spectrogram;
- 128 mel bins;
- 25 ms Hamming window;
- 10 ms hop;
- standard normalization.

Для baseline:

- MFCC 40 коэффициентов + delta + delta-delta;
- агрегировать mean/std по времени;
- SVM/logistic regression.

### Аугментации

Базовый набор:

- random crop/cyclic padding;
- additive noise на умеренном SNR;
- time shift;
- time stretching без сильного искажения;
- SpecAugment: time masking + frequency masking;
- mixup или patch-mix;
- class-balanced sampler или weighted loss.

Не стоит агрессивно использовать pitch shift: для дыхания pitch не совпадает с речевым сценарием, и можно исказить клинически значимые частоты. Его можно оставить как ablation, а не как основной метод.

### Дисбаланс

ICBHI сильно несбалансирован. Особенно мало класса `both`. Для обучения:

- weighted cross entropy;
- balanced sampling;
- macro F1 и per-class recall в дополнение к ICBHI Score;
- confusion matrix обязательно.

Для диплома важно показывать не только итоговый `Score`, но и какие классы модель путает. Типичная проблема: `both` часто путается с `crackle` или `wheeze`.

## Математическая постановка

Пусть есть набор пациентов `P`, набор аудиозаписей и дыхательных циклов:

```text
D = {(x_i, y_i, p_i)}_{i=1}^N
```

где:

- `x_i` - аудиосигнал дыхательного цикла;
- `y_i in {normal, crackle, wheeze, both}`;
- `p_i` - идентификатор пациента.

Модель:

```text
f_theta: x_i -> p_theta(y | x_i)
```

После предобработки:

```text
X_i = LogMel(x_i)
f_theta(X_i) = softmax(g_theta(X_i))
```

Оптимизация:

```text
theta* = argmin_theta (1/N) sum_i w_{y_i} * CE(y_i, f_theta(X_i))
```

Для SAM-Optimized AST:

```text
L_SAM(theta) = max_{||epsilon|| <= rho} L(theta + epsilon)
theta* = argmin_theta L_SAM(theta)
```

Для ADD-RSC baseline:

```text
L = beta * L_bias_denoise + (1 - beta) * L_CE
```

Оценка:

```text
Sp = TP_normal / N_normal
Se = (TP_crackle + TP_wheeze + TP_both) / (N_crackle + N_wheeze + N_both)
Score = (Sp + Se) / 2
```

Дополнительные метрики:

- accuracy;
- macro F1;
- per-class recall;
- confusion matrix;
- AUROC для binary normal/abnormal;
- confidence intervals по нескольким seed.

## Практическая рекомендация для диплома

Минимальный, но сильный набор экспериментов:

1. `MFCC/log-mel statistics + SVM/logistic regression` как классический baseline.
2. `ResNet18/ResNet50 on log-mel` как CNN baseline.
3. `AST fine-tuning` как transformer baseline.
4. `ADD-RSC on AST` как сильный denoising baseline.
5. `SAM-Optimized AST` как выбранное современное решение на 2026 год.

Основной результат:

- ICBHI official split;
- 4-class classification;
- reporting: Sp, Se, Score, macro F1, confusion matrix;
- 3-5 random seeds, если хватает времени.

Что показать в 3-й главе:

- таблица конфигов;
- графики train/val loss;
- итоговая таблица метрик;
- confusion matrix;
- анализ ошибок по классам;
- анализ влияния preprocessing/augmentation;
- вывод, почему high specificity легче, чем high sensitivity;
- ограничения: размер датасета, дисбаланс, шум, разные стетоскопы, отсутствие клинической валидации.

## Источники

- WHO, Chronic respiratory diseases: https://www.who.int/europe/news-room/fact-sheets/item/chronic-respiratory-diseases
- WHO, COPD fact sheet: https://www.who.int/en/news-room/fact-sheets/detail/chronic-obstructive-pulmonary-disease-%28copd%29
- ICBHI 2017 Challenge official dataset: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- ICBHI challenge welcome/task description: https://bhichallenge.med.auth.gr/welcome
- Yu et al., 2025, "Advances and Challenges in Respiratory Sound Analysis: A Technique Review Based on the ICBHI2017 Database": https://www.mdpi.com/2079-9292/14/14/2794
- Gong et al., 2021, "AST: Audio Spectrogram Transformer": https://arxiv.org/abs/2104.01778
- Chen et al., 2023, "BEATs: Audio Pre-Training with Acoustic Tokenizers": https://proceedings.mlr.press/v202/chen23ag.html
- Bae et al., 2023, "Patch-Mix Contrastive Learning with Audio Spectrogram Transformer on Respiratory Sound Classification": https://arxiv.org/abs/2305.14032
- Xiao et al., 2024, "LungAdapter: Efficient Adapting Audio Spectrogram Transformer for Lung Sound Classification": https://www.isca-archive.org/interspeech_2024/xiao24_interspeech.html
- Dong et al., 2025, "Adaptive Differential Denoising for Respiratory Sounds Classification": https://www.isca-archive.org/interspeech_2025/dong25e_interspeech.pdf
- Isik et al., 2025/2026 frontier, "Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers": https://arxiv.org/abs/2512.22564
- Kim et al., 2026, "Meta-Ensemble Learning with Diverse Data Splits for Improved Respiratory Sound Classification": https://arxiv.org/abs/2604.24096
- Zhang et al., 2026, "DPAST-LS: A transformer-based self-distillation network for efficient recognition of abnormal pulmonary sounds": https://www.sciencedirect.com/science/article/abs/pii/S1746809425017859
- K.V. and Anand, 2026, "Investigation into respiratory sound classification for an imbalanced data set using hybrid LSTM-KAN architectures": https://www.sciencedirect.com/science/article/pii/S2666990025000527
- Akbasli and Serin, 2026, "PulmoVec: A Two-Stage Stacking Meta-Learning Architecture Built on the HeAR Foundation Model": https://arxiv.org/abs/2603.15688
- Google HeAR model card: https://huggingface.co/google/hear
- Kim et al., 2026, "Respiratory sound analysis for ICU clinical decision support": https://link.springer.com/article/10.1186/s12911-026-03409-0
- Zhang et al., 2022, SPRSound repository: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- HF_Lung_V1 GitLab: https://gitlab.com/techsupportHF/HF_Lung_V1
- Hsu et al., 2021, HF_Lung_V1 paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8248710/
- RespiratoryDatabase@TR on Mendeley Data: https://data.mendeley.com/datasets/p9z4h98s6j/1
- Coswara Data repository: https://github.com/iiscleap/Coswara-Data
- OPERA, 2024, "Towards Open Respiratory Acoustic Foundation": https://proceedings.neurips.cc/paper_files/paper/2024/file/2f803abdcad9de35b45d5a656dade45c-Paper-Datasets_and_Benchmarks_Track.pdf
- SpecAugment, 2019: https://research.google/pubs/specaugment-a-simple-augmentation-method-for-automatic-speech-recognition/
- PCEN overview/paper page: https://www.lostanlen.com/pubs/lostanlen2019spl/
- mixup, 2018: https://arxiv.org/abs/1710.09412
