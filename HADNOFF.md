# Handoff

Дата: 2026-06-08.

Последний запрос пользователя: сделать валидный `.gitignore`, в `report/` оставить только PDF, остальное игнорировать, также игнорировать большие файлы с данными и похожие артефакты.

Сделано:

- 2026-06-08 по последнему запросу пользователя создан корневой `.gitignore`:
  - `report/**` игнорируется полностью, кроме PDF-файлов внутри `report`;
  - добавлены правила для OS/editor мусора, Python cache/env, локальных логов, Mutagen lock;
  - добавлены правила для больших данных, датасетов, run-артефактов, архивов, бинарных дампов, моделей и checkpoint-файлов.
  - Тесты и серверные команды не запускались, потому что задача касалась только ignore-правил.
- 2026-06-08 добавлена train-time аугментация в `src/data/dataset.py`:
  - waveform-level: random gain, circular time shift, Gaussian noise;
  - log-mel-level: frequency mask и time mask;
  - включается через `training.augmentation` и применяется только к train split, опционально по списку label.
- `src/train.py` передает параметры аугментации в `RespiratoryAudioDataset`.
- Добавлены конфиги:
  - `configs/experiments/ast_base_unbalanced.yaml`;
  - `configs/experiments/ast_base_augmented.yaml`.
- `scripts/train_queue.sh` расширен AST-запусками: unbalanced, weighted, focal, augmented.
- Обновлены карточки экспериментов:
  - `EXPS/006_ast_base_focal.md` теперь описывает подготовленный focal-loss метод;
  - `EXPS/007_ast_base_unbalanced.md` создан и заполнен фактическими метриками;
  - `EXPS/008_ast_base_augmented.md` создан для аугментационного режима.
- На `JH` выполнен `ruff check src configs scripts`, ошибок нет.
- На `JH` запущен `ast_base_unbalanced`; полный 50-эпоховый запуск был остановлен после 29-й эпохи из-за отсутствия улучшения после лучшей 22-й эпохи. Сохраненный `best.pt` оценен через `src.evaluate`, `test_metrics_recomputed.json` скопирован в стандартный `test_metrics.json`.
- Фактический run: `/home/margasanov/temki/diplom/reports/runs/20260608_131349_ast_base_unbalanced`.
- Метрики unbalanced AST: accuracy `0.5000`, macro F1 `0.4657`, weighted F1 `0.5181`, balanced accuracy `0.5029`; recall: normal `0.4597`, crackle `0.7754`, wheeze `0.3581`, both `0.4184`.
- Обновлены `report/chapters/chapter3.tex` и `report/chapters/conclusion.tex`:
  - добавлен контрольный AST без балансировки;
  - добавлен раздел `Работа с несбалансированной выборкой`;
  - старый блок про соотношение сложности и качества удален, чтобы объем отчета не рос;
  - добавлено сравнение unbalanced AST и weighted AST.
- Обновлен `README.md` с актуальными AST-командами, аугментацией и текущими результатами.

Осталось:

- Пересобрать `report/main.pdf` после последних LaTeX-правок.
- Запустить `src.report --runs-dir reports/runs --output reports/summary.csv` на `JH`, если нужен обновленный `reports/summary.csv` с unbalanced AST.
- `ast_base_focal` и `ast_base_augmented` подготовлены, но не обучены в этом проходе из-за длительности AST fine-tuning.

Ранее сделано:

- 2026-06-08 создан `DATASETS.md` с исследованием дополнительных источников данных для текущей задачи `normal / crackle / wheeze / both`:
  - приоритет 1: `SPRSound`, потому что есть WAV, JSON event-level разметка и классы, близкие к ICBHI;
  - приоритет 2: `HF_Lung_V1`, потому что много lung sound файлов и разметки wheeze/crackle, но лучше подключать через multi-label или auxiliary-задачу;
  - приоритет 3: Fraiwan / KAUH с Mendeley, небольшой stethoscope-набор с `W`, `C`, `N`;
  - дополнительные источники для pretraining/noise/hard negatives: `RespiratoryDatabase@TR`, `Coswara`, `Sound-Dr`, `AudioSet respiratory sounds`.
- В `DATASETS.md` добавлены риски прямого смешивания источников, схема конвертации меток, рекомендация держать чистый ICBHI test и добавить колонку `dataset` в `index.csv`.
- Тесты и проектные команды не запускались, потому что пользователь просил исследование и документацию, а локальные правила запрещают тестовую работу без прямого запроса.

Ранее сделано:

- 2026-06-08 расширена `report/chapters/chapter3.tex` с 254 строк до 671 строки:
  - добавлены блоки про динамику обучения всех нейросетевых моделей, сводка лучших checkpoint по валидации, сравнение validation macro F1 по эпохам, отдельные графики для ResNet18 и CRNN GRU;
  - добавлены таблицы и графики по пер-классовым метрикам AST-Base, сравнении F1-score по классам между всеми моделями и разрыву между лучшей валидацией и тестом;
  - добавлен блок про соотношение сложности и качества с таблицей параметров и scatter plot по числу параметров против качества.
- `report/main.pdf` пересобран через локальный временный Tectonic из `/private/tmp` и синхронизирован через `mutagen project flush`.
- Проверка `report/main.log`: глава 3 начинается на странице 42, глава 4 (заключение) начинается на странице 58, то есть глава 3 занимает 16 страниц и попадает в требуемый диапазон; незакрытых `undefined reference/citation` и `LaTeX Error` нет.
- 2026-06-08 расширена `report/chapters/chapter2.tex` с 254 до 372 строк:
  - добавлены разделы про требования к входному аудиофрагменту, акустические признаки целевых классов, адаптацию pretrained AST-Base, регуляризацию/аугментации спектрограмм, организацию обучения и валидации;
  - добавлена таблица параметров формирования log-mel спектрограммы;
  - расширены объяснения patch embedding, позиционных вложений, самовнимания, классификационного токена, дисбаланса классов и ограничений AST-Base.
- `report/main.pdf` пересобран локальным временным Tectonic из `/private/tmp` и синхронизирован через `mutagen project flush`.
- Проверка `report/main.log`: глава 2 начинается на странице 24, глава 3 начинается на странице 42, то есть глава 2 занимает 18 страниц и попадает в диапазон 15-20 страниц; незакрытых `undefined reference/citation` и `LaTeX Error` нет.
- 2026-06-08 в `report/main.tex` исправлены горизонтальные поля PDF: вместо `left=30mm,right=15mm` задано `left=25mm,right=25mm`.
- `report/main.pdf` пересобран через локальный временный Tectonic из `/private/tmp`, потому что на `JH` не найдены `tectonic`, `xelatex` и `latexmk`; после сборки выполнен `mutagen project flush`, сервер видит обновленный PDF.
- Проверка `report/main.log`: `h-part:(L,W,R)=(71.13188pt, 455.24411pt, 71.13188pt)`, то есть левое и правое поля одинаковые.
- 2026-06-08 расширена `report/chapters/chapter1.tex`:
  - раздел 1.2 значительно увеличен: добавлены подробные абзацы про ручные признаки, CNN/ResNet, CRNN, AST, preprocessing, аугментации, transfer learning, ансамбли, облегченные модели, протоколы оценки и итоговое обоснование AST-Base;
  - раздел 1.3 умеренно расширен: добавлены patient-independent split, различие файлов и сегментов, согласование классов, качество аннотаций и шум;
  - раздел 1.6 немного расширен: добавлены precision/recall, macro F1 для checkpoint selection, ограничения weighted F1 и связь метрик с графиками обучения;
  - PDF пересобран: по `report/main.log` глава 1 начинается на странице 4, глава 2 начинается после страницы 23, то есть глава 1 занимает около 20 страниц.
- `DIPLOM_STRUCTURE.md` обновлен под увеличенную главу 1 и новые акценты в разделах 1.2, 1.3 и 1.6.
- 2026-06-08 обновлен отчет и пересобран `report/main.pdf`:
  - заголовки глав, разделов и подразделов уменьшены сильнее через `titlesec`;
  - в список литературы добавлены кликабельные `\href{...}{...}` DOI/arXiv/URL;
  - все упоминания `86191876` заменены на `86м параметров`; график числа параметров AST-Base также округлен до `86`;
  - в главу 3 добавлены графики обучения AST-Base: train/validation loss и validation macro F1/balanced accuracy по эпохам;
  - в главу 3 добавлен расширенный анализ результатов: интерпретация кривых обучения, переобучения, различия weighted F1 и macro F1, ограничений класса `both` и дальнейших экспериментов.
- Попытка скопировать PNG-графики с `JH` через `scp`/`ssh cat` зависла; зависшие процессы были остановлены. Вместо PNG использованы реальные точки из серверного `metrics.csv`, полученные через `ssh` и построенные в LaTeX через `pgfplots`.
- Финальная проверка: `report/main.pdf` валидный PDF, `report/main.log` не содержит незакрытых `undefined reference/citation` или `error`.
- 2026-06-08 `DIPLOM_STRUCTURE.md` полностью обновлен под текущий отчет:
  - добавлены правила оформления про компактные заголовки и обязательные ссылки на нумерованные элементы;
  - глава 2 теперь явно сфокусирована на `AST-Base`, log-mel preprocessing, patch embedding, attention, transformer block, classifier head, transfer learning и дисбалансе;
  - старое требование подробно описывать ручные признаки, CNN/ResNet и CRNN в главе 2 убрано;
  - в структуру добавлены обязательные рисунки для главы 2 и графики результатов для главы 3;
  - в структуру внесены фактические метрики и per-class recall из завершенных экспериментов.
- 2026-06-08 по новому запросу пользователя обновлен отчет и пересобран `report/main.pdf`:
  - в `report/main.tex` добавлены `titlesec`, `tikz`, `pgfplots`, уменьшены заголовки глав, разделов и подразделов;
  - глава 2 полностью переработана: убраны подробные описания невыбранных методов, фокус перенесен на AST-Base, log-mel preprocessing, patch embedding, self-attention, transformer block, classifier head, transfer learning и учет дисбаланса;
  - в главу 2 добавлены схемы: конвейер AST-Base, схематическая log-mel спектрограмма, разбиение на патчи, блок трансформера, схема учета дисбаланса;
  - в главу 3 добавлены графики: распределение классов, число параметров моделей, сравнение агрегированных метрик, сравнение recall по классам;
  - убраны длинные пути и длинный checkpoint из основного текста, чтобы уменьшить проблемы верстки.
- Финальная сборка `report/main.pdf` прошла успешно через Tectonic; `report/main.log` не содержит незакрытых `undefined reference/citation` или `error`.
- 2026-06-08 собран готовый PDF: `report/main.pdf`.
- Для сборки скачан временный `tectonic 0.16.9` в `/private/tmp/tectonic-0.16.9`; полноценного LaTeX toolchain на `JH` и локальной машине не было.
- `report/main.tex` переведен с `fontenc/inputenc` на XeTeX-compatible `fontspec`, потому что Tectonic/XeTeX упал на T2A-шрифте `larm1440`. Используются системные шрифты `Times New Roman`, `Arial`, `Courier New`.
- Финальная сборка Tectonic прошла успешно, создан `report/main.pdf` и `report/main.log`. В `report/main.log` нет незакрытых `undefined reference/citation`; остались только предупреждения о overfull/underfull hbox и предупреждения о системных шрифтах.
- 2026-06-08 создана папка `report/` с LaTeX-исходниками итогового отчета:
  - `report/main.tex` как основной входной файл;
  - `report/chapters/introduction.tex`;
  - `report/chapters/chapter1.tex`;
  - `report/chapters/chapter2.tex`;
  - `report/chapters/chapter3.tex`;
  - `report/chapters/conclusion.tex`;
  - `report/chapters/references.tex`;
  - `report/chapters/appendix.tex`;
  - `report/README.md`.
- Отчет написан по структуре `DIPLOM_STRUCTURE.md` и привязан к фактическому состоянию проекта: использованы данные из `README.md`, `RESEARCH.md`, `EXPS/001_prepare_data.md` ... `EXPS/006_ast_base_focal.md`, конфигов и модулей `src/`.
- В отчете зафиксированы реальные результаты завершенных экспериментов: MFCC + logistic regression, ResNet18, CRNN GRU и AST-Base. `ast_base_focal` явно не включен в итоговое сравнение как незапущенная ablation-конфигурация.
- PDF не собирался: пользователь просил LaTeX, а текущая задача была созданием исходников. Тесты не запускались, потому что локальные правила запрещают test-related work без прямой просьбы пользователя.
- 2026-06-08 в `AGENTS.md` добавлены правила написания текста диплома: безличный академический стиль, минимум ненужных англицизмов, обязательные ссылки в тексте на нумерованные формулы/рисунки/таблицы/листинги, единая терминология и стилистика. Тесты и серверные команды не запускались, потому что правка только документационная.
- 2026-06-08 в `AGENTS.md` добавлен раздел `Diploma Report Generation`: отчет генерируется как LaTeX-исходники, Python используется только для воспроизводимых фрагментов, итоговый PDF собирается LaTeX toolchain из `main.tex`. Тесты и серверные команды не запускались, потому что правка только документационная.
- 2026-06-08 в `DIPLOM_STRUCTURE.md` расширен раздел `1.1. Прикладной и технический контекст`: добавлены пункты про актуальность задачи, прикладную постановку, техническую сложность и причины использовать современные audio/transfer learning модели. Тесты и серверные команды не запускались, потому что правка только документационная.
- 2026-06-08 обновлены итоговые карточки в `EXPS/001_prepare_data.md` ... `EXPS/006_ast_base_focal.md`: добавлены размеры датасета, run paths, test metrics, per-class recall, сравнение методов и выводы. `006_ast_base_focal` явно отмечен как не запущенная ablation-карточка.
- 2026-06-08 проверено: `JH` доступен, Mutagen `diplom` подключен с обеих сторон, `mutagen project flush` проходит.
- Очередь обучения завершилась полностью; активных `src.train`, `src.report`, `train_queue.sh` процессов нет.
- GPU свободна: H100 показывает `0 MiB` занято и `0%` utilization после завершения.
- Серверные результаты лежат в `/home/margasanov/temki/diplom/reports`; размер `reports` около `383M`.
- `reports/summary.csv` и `reports/summary.json` созданы на сервере.
- Финальные результаты `reports/summary.csv`:
  - `ml_mfcc_logreg`: macro F1 `0.2878`, balanced accuracy `0.2882`, weighted F1 `0.4325`;
  - `resnet18_logmel`: macro F1 `0.3044`, balanced accuracy `0.3396`, weighted F1 `0.3865`;
  - `crnn_gru_logmel`: macro F1 `0.2998`, balanced accuracy `0.3062`, weighted F1 `0.4080`;
  - `ast_base`: macro F1 `0.4867`, balanced accuracy `0.4928`, weighted F1 `0.5817`.
- Лучший run: `20260607_091747_ast_base`; test accuracy `0.5776`, per-class recall: normal `0.6619`, crackle `0.5241`, wheeze `0.5608`, both `0.2245`.
- Из-за `mutagen.yml` mode `one-way-safe` серверные `reports/` не синхронизируются обратно локально; читать результаты нужно на `JH` или отдельно копировать вручную при необходимости.
- 2026-06-07 обучение поставлено на `JH` через `nohup bash scripts/train_queue.sh > reports/jobs/train_queue.log 2>&1 &`.
- Активная очередь подтверждена: `train_queue.sh` PID `6458`, текущий этап `ml_mfcc_logreg` через `python -m src.train --config configs/experiments/ml_mfcc_logreg.yaml`.
- `src/data/audio.py` исправлен: `torchaudio.load` заменен на `soundfile.read` + `librosa.resample`, потому что первый запуск упал на `torchcodec/ffmpeg` shared libraries.
- Проверено на `JH`: `python -m compileall src/data/audio.py` проходит, один WAV загружается как waveform shape `(320000,)`, dtype `torch.float32`.
- Датасет распакован в `/opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database`; index build сохранил `6898` строк в `data/processed/index.csv`.
- Текущий лог обучения: `/home/margasanov/temki/diplom/reports/jobs/train_queue.log`.
- Создана структура практической репы: `configs/`, `src/`, `scripts/`, `notebooks/model_research/`, `EXPS/`.
- Реализован data pipeline:
  - `src/data/index.py` собирает `data/processed/index.csv`;
  - поддержан ICBHI-style txt format `start end crackle wheeze`;
  - split делается patient-independent через `patient_id`, если возможно;
  - audio приводится к 16 kHz mono в loaders.
- Реализованы фичи:
  - MFCC + delta + delta-delta + статистики;
  - spectral centroid/bandwidth/rolloff, RMS, ZCR;
  - log-mel spectrogram для CNN/CRNN/AST.
- Реализованы модели:
  - classical ML: logistic regression, SVM, random forest;
  - ResNet18/ResNet50 на log-mel;
  - CRNN с CNN encoder и GRU/LSTM;
  - AST через Hugging Face `ASTForAudioClassification` с `local_files_only=True`.
- Реализован `src/train.py`:
  - ML и torch training;
  - weighted sampler;
  - weighted CE и focal loss;
  - сохранение `config.yaml`, `model_info.json`, `metrics.csv`, `test_metrics.json`, confusion matrix, training curves, best checkpoint.
- Реализованы `src/evaluate.py` и `src/report.py`.
- Добавлены конфиги экспериментов:
  - `ml_mfcc_logreg.yaml`;
  - `resnet18.yaml`;
  - `crnn_gru.yaml`;
  - `ast_base.yaml`;
  - `ast_base_focal.yaml`.
- Добавлены md-карточки экспериментов в `EXPS/001_prepare_data.md` ... `EXPS/006_ast_base_focal.md`.
- Добавлены notebooks в `notebooks/model_research/`:
  - `01_classical_ml.ipynb`;
  - `02_cnn_crnn_research.ipynb`;
  - `03_ast_research.ipynb`.
- Добавлен `scripts/download_hf_assets.py` для загрузки HF assets на машине с интернетом.
- Обновлен `README.md` с командами запуска, модульным обзором и текущими ограничениями.
- На `JH` установлены недостающие зависимости в conda env `/opt/gen-content/margasanov/envs/pizding-kartocheck`: `librosa`, `scikit-learn`, `seaborn`, `joblib`, `ruff` и их зависимости через разрешенный PyPI artifactory.
- На `JH` проверено:
  - `python -m compileall src scripts` проходит;
  - `python -m ruff check .` проходит;
  - CUDA доступна: `torch 2.10.0+cu128`, `torch.cuda.is_available() == True`;
  - `torchaudio 2.10.0+cu128`, `torchvision 0.25.0+cu128`, `librosa 0.11.0`, `transformers 5.9.0`;
  - `ASTForAudioClassification` импортируется;
  - ResNet18 создается, параметров `11172292`;
  - CRNN создается, параметров `2063300`.
- AST assets скачаны на `JH` в `/opt/gen-content/margasanov/hf_models/ast-finetuned-audioset-10-10-0.4593`.
- AST модель грузится с `local_files_only=True`; параметров `86191876`; classifier head штатно пересоздается с 527 AudioSet классов на 4 класса.
- ICBHI 2017 official archive скачан локально из `https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip` в `/private/tmp/icbhi_download/ICBHI_final_database.zip`.
- Локальный zip проверен через `unzip -t`, ошибок нет.
- Архив передан на JH в `/opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database.zip`.
- Серверный zip проверен через `unzip -t`, ошибок нет; размер `1978998275` bytes.
- Распаковка на JH завершена; в `/opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database` есть WAV-файлы датасета.
- `configs/data/default.yaml` теперь указывает `raw_dir: /opt/gen-content/margasanov/datasets/icbhi/ICBHI_final_database`.
- `src/data/index.py` больше не размечает wav без txt-аннотации как `normal`; такие файлы пропускаются.
- Добавлен `scripts/train_queue.sh`: последовательная очередь `index -> ml_mfcc_logreg -> resnet18 -> crnn_gru -> ast_base -> report`.
- Первая попытка запуска очереди собрала index, но упала на `torchaudio.load` из-за `torchcodec/ffmpeg`; после фикса `src/data/audio.py` очередь перезапущена.

Ранее сделано:

- `README.md`, `PLAN.md` и `DIPLOM_STRUCTURE.md` переписаны под новую логику проекта: сравнение 4 семейств моделей, выбор лучшей базовой архитектуры и domain-specific fine-tuning.
- Выбран основной подход: `AST-Base`, потому что transformer-подход использует pretrained audio representation, работает с log-mel spectrogram и укладывается в лимит `100M` параметров; в документах указан ориентир порядка `86M` параметров.
- Метрики заменены на общие для выбранного датасета: macro F1, balanced accuracy, per-class recall, weighted F1 и confusion matrix.
- Старый фокус на `SAM-Optimized AST`, `ADD-RSC`, `ICBHI Score`, `Sp/Se` и SOTA-сравнении убран из `README.md`, `PLAN.md`, `DIPLOM_STRUCTURE.md`.
- Основной датасет теперь описан как raw audio dataset около `5GB`; ICBHI оставлен только как возможный reference benchmark в структуре диплома.
- В `PLAN.md` задача переформулирована как классификация дыхательных аудиосегментов и экспериментальный audio pipeline; убраны выводы про clinical validation/скрининг.
- В `DIPLOM_STRUCTURE.md` введение и глава 1 переписаны в техническую сторону: ICBHI-разметка, audio classification, small/noisy/imbalanced dataset, transfer learning.
- Убраны явные клинические маркеры из `PLAN.md` и `DIPLOM_STRUCTURE.md`: `медицинский`, `клинический`, `скрининг`, `diagnosis/clinical`, `COPD/asthma/pneumonia`, `аускультация`.
- Добавлен `mutagen.yml` для синхронизации локального проекта `/Users/margasanov/bmstu/8sem/diplom` в `/home/margasanov/temki/diplom` на `JH`.
- Запущена Mutagen project-сессия `diplom`; состояние `Watching for changes`, `Alpha` и `Beta` подключены.
- Выполнен `mutagen project flush`; файлы проекта появились на сервере.
- Проверена conda-среда `/opt/gen-content/margasanov/envs/pizding-kartocheck`; Python отвечает как `3.11.15`.
- Обновлен `README.md` с серверной инструкцией по Mutagen и runtime environment.
- Создан `RESEARCH.md` с обзором задачи, архитектур, датасетов, обработки данных, метрик и источников.
- Создан `PLAN.md` с практическим планом реализации.
- Создан `DIPLOM_STRUCTURE.md` со структурой диплома.
- Создан `README.md`, потому что локальные инструкции требуют основной документ проекта.
- Создан этот `HADNOFF.md`.
- В `AGENTS.md` добавлено правило держать код простым, не добавлять лишние функции/проверки/абстракции, разделять логику по focused files и смысловым блокам, держать докстринги короткими.
- В `AGENTS.md` добавлено правило про лимит `/home/margasanov` в `20GB` и размещение больших файлов, датасетов, весов моделей, кешей и тяжелых артефактов в `/opt/gen-content/margasanov`.

Ключевое текущее решение: основная практическая модель - `AST-Base` как лучший transformer-подход при лимите `100M` параметров. Практика должна показать сравнение с ручными признаками, CNN/ResNet и CRNN, затем обучить выбранную `AST-Base` на domain-specific задаче.

Важно:

- Текущий `AGENTS.md` просит игнорировать git в этой директории.
- Проектные команды нужно выполнять на `JH` после `mutagen project flush` и в conda environment `/opt/gen-content/margasanov/envs/pizding-kartocheck`.
- Тесты не создавались и не запускались, потому что текущие локальные правила запрещают test-related work без прямой просьбы пользователя.
- Перед реальным запуском нужно после восстановления SSH к JH выполнить:
  - `cd /opt/gen-content/margasanov/datasets/icbhi && unzip -q -o ICBHI_final_database.zip`;
  - `cd /home/margasanov/temki/diplom && conda run -p /opt/gen-content/margasanov/envs/pizding-kartocheck python -m src.data.index --config configs/data/default.yaml`;
  - `cd /home/margasanov/temki/diplom && mkdir -p reports/jobs && nohup bash scripts/train_queue.sh > reports/jobs/train_queue.log 2>&1 & echo $!`.
