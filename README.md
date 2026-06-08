# Дипломный проект

Тема: "Прогнозирование респираторной болезни на основе анализа звукозаписи дыхания".

Проект содержит практическую реализацию pipeline для сравнения нескольких семейств моделей классификации дыхательных аудиозаписей, выбора базовой архитектуры до `100M` параметров и domain-specific обучения на сыром датасете около `5GB`.

## Документы

- `RESEARCH.md` - подробный ресерч по задаче, датасетам, классам моделей, бенчмаркам и выбранной архитектуре.
- `PLAN.md` - план практической реализации экспериментов.
- `DIPLOM_STRUCTURE.md` - верхнеуровневая структура дипломной работы по главам.
- `HADNOFF.md` - краткий handoff для следующего агента.

## Цель проекта

Практическая цель - построить воспроизводимый pipeline:

1. Подготовить сырой аудиодатасет объемом около `5GB`.
2. Реализовать baseline на ручных признаках.
3. Реализовать CNN/ResNet baseline на спектрограммах.
4. Реализовать CRNN baseline: CNN + RNN/LSTM/GRU.
5. Реализовать transformer baseline и выбрать лучшую базовую архитектуру.
6. Обучить выбранную архитектуру на domain-specific задаче.
7. Оценить качество по macro F1, balanced accuracy, per-class recall и confusion matrix.
8. Проверить влияние несбалансированной выборки и подготовить методы балансировки, focal loss и аугментации меньших классов.

## Локальный старт

Локальная машина используется для разработки и синхронизации. GPU-запуск выполняется только на `JH`.

1. Разместить сырой датасет в `/opt/gen-content/margasanov/datasets/` на `JH`.
2. Если нужен AST, заранее загрузить веса на машине с интернетом и перенести их на сервер в:

```text
/opt/gen-content/margasanov/hf_models/ast-finetuned-audioset-10-10-0.4593
```

3. Синхронизировать проект:

```bash
mutagen daemon start
mutagen project start
mutagen project flush
```

## Серверная синхронизация

Проект синхронизируется на `JH` через Mutagen из локального корня:

```bash
mutagen daemon start
mutagen project start
mutagen project flush
```

Рабочий путь на сервере: `/home/margasanov/temki/diplom`.

Все будущие проектные команды нужно запускать на `JH` после `mutagen project flush` и только в conda environment `/opt/gen-content/margasanov/envs/pizding-kartocheck`.

## Команды

Подготовить index table:

```bash
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.data.index --config configs/data/default.yaml'
```

Запустить baseline на ручных признаках:

```bash
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ml_mfcc_logreg.yaml'
```

Запустить torch-эксперименты:

```bash
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/resnet18.yaml'
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/crnn_gru.yaml'
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_unbalanced.yaml'
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base.yaml'
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_focal.yaml'
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.train --config configs/experiments/ast_base_augmented.yaml'
```

Собрать таблицу завершенных запусков:

```bash
ssh JH 'cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.report --runs-dir reports/runs --output reports/summary.csv'
```

## Модульный обзор реализации

- `src/data/` - чтение WAV, парсинг ICBHI-аннотаций, fixed-length сегменты, train-time аугментации и сбор `index.csv`.
- `src/features/` - MFCC/statistics и log-mel spectrogram.
- `src/models/` - ML baseline, ResNet, CRNN и AST-Base.
- `src/train.py` - обучение ML и torch-моделей с логированием конфигов, метрик, чекпоинтов и графиков.
- `src/evaluate.py` - повторная оценка сохраненного torch-checkpoint.
- `src/report.py` - агрегация метрик из `reports/runs`.
- `configs/` - параметры данных и экспериментов.
- `EXPS/` - отдельные markdown-карточки экспериментов.
- `notebooks/model_research/` - Jupyter notebooks для ресерча остальных моделей без обязательного интерактивного прогона.

## Текущее состояние запуска

На `JH` проверено: CUDA доступна, импорты `torch`, `torchaudio`, `torchvision`, `librosa`, `transformers` проходят, `ruff check src configs scripts` проходит.

AST-веса скачаны в `/opt/gen-content/margasanov/hf_models/ast-finetuned-audioset-10-10-0.4593`; модель грузится с `local_files_only=True`, параметров `86191876`.

ICBHI 2017 скачан локально из official challenge URL в `/private/tmp/icbhi_download/ICBHI_final_database.zip`, проверен через `unzip -t`, передан на JH в `/opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database.zip` и проверен на JH через `unzip -t`.

Завершены и зафиксированы в `EXPS/` основные эксперименты: MFCC + logistic regression, ResNet18, CRNN GRU, AST-Base с балансировкой и контрольный AST-Base без балансировки. Основной результат: сбалансированная AST-Base получает accuracy `0.5776`, macro F1 `0.4867`, weighted F1 `0.5817`, balanced accuracy `0.4928`. Контрольный AST-Base без балансировки получает macro F1 `0.4657`, weighted F1 `0.5181`, balanced accuracy `0.5029`.

Для дальнейшей проверки добавлены конфиги `ast_base_focal.yaml` и `ast_base_augmented.yaml`. Аугментация применяется только в train split для классов `crackle`, `wheeze`, `both`: random gain, time shift, Gaussian noise, frequency mask и time mask.
