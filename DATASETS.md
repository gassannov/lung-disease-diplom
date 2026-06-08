# DATASETS.md

Дата исследования: 2026-06-08.

## Проблема

Текущий практический датасет ICBHI 2017 мал для обучения нейросетевых моделей с миллионами параметров. В проекте уже используется сегментация на `6898` дыхательных циклов, но независимых пациентов всего `126`, а классы `wheeze` и особенно `both` представлены заметно хуже класса `normal`. Такая структура объясняет быстрое переобучение: модель много раз видит близкие сегменты одних и тех же пациентов, быстро запоминает акустические условия записи и хуже переносится на валидацию.

Добавлять данные можно, но не все открытые "respiratory sound" наборы одинаково полезны. Для текущей задачи `normal / crackle / wheeze / both` наиболее ценны именно записи с грудной клетки и event-level разметкой хрипов. Кашель, речь и дыхание через микрофон телефона полезны слабее: они могут помочь для предобучения или устойчивости к шуму, но напрямую смешивать их с ICBHI в supervised-классификацию нельзя без изменения постановки задачи.

## Короткий вывод

Рекомендуемый порядок действий:

1. Добавить `SPRSound` как самый полезный открытый источник с event-level разметкой `Normal`, `Wheeze`, `Coarse Crackle`, `Fine Crackle`, `Wheeze+Crackle`, `Rhonchi`, `Stridor`.
2. Добавить `HF_Lung_V1` для расширения событий wheeze/crackle и для обучения детектора "есть добавочный звук / нет добавочного звука".
3. Добавить датасет Fraiwan / KAUH с Mendeley как небольшой, но совместимый источник stethoscope-записей с метками `W`, `C`, `N`.
4. Использовать `Coswara`, `Sound-Dr`, `CoughVID` и `AudioSet Respiratory sounds` не как прямое расширение классов ICBHI, а как предобучение, hard negatives, шумовую регуляризацию и auxiliary-задачи.
5. При объединении датасетов перейти от "слить все сегменты в один train" к `source-aware` протоколу: отдельная колонка `dataset`, patient-independent split внутри источников и отдельный тест на чистом ICBHI.

## Кандидаты для добавления

| Приоритет | Датасет | Польза для проекта | Совместимость с текущими классами | Риск |
|---|---|---|---|---|
| 1 | SPRSound | Существенно больше pediatric respiratory events, есть event-level JSON | Высокая: `Normal`, `Wheeze`, `Coarse/Fine Crackle`, `Wheeze+Crackle` можно привести к `normal/crackle/wheeze/both` | Другая возрастная группа и другой стетоскоп |
| 2 | HF_Lung_V1 | Очень много 15-секундных lung sound файлов и разметки CAS/DAS | Средняя: wheeze/rhonchus/stridor -> CAS, crackles -> DAS; для 4 классов нужна аккуратная конвертация | Не та же схема классов, нет прямого `both` как в ICBHI |
| 3 | Fraiwan / KAUH lung sounds | Небольшой stethoscope-набор с нормой, wheeze, crackle и диагнозами | Средняя/высокая: `W`, `C`, `N`; combined события нужно проверять | Мало данных, разные частоты и фильтры стетоскопа |
| 4 | RespiratoryDatabase@TR | 12-канальные lung sounds, COPD severity, CC BY 4.0 | Низкая/средняя: больше подходит для disease/severity или domain pretraining | Нет прямой event-разметки ICBHI-классов |
| 5 | Coswara | Много дыхания/кашля/речи, CC BY 4.0 | Низкая для stethoscope-классов, высокая для pretraining | Запись телефоном, другой сигнал |
| 6 | Sound-Dr | Кашель, mouth/nose breathing, метаданные заболеваний | Низкая для crackle/wheeze, полезно для предобучения | Не auscultation, лицензии/доступ нужно проверить |
| 7 | CoughVID | Большой cough corpus с экспертной частью разметки | Низкая для crackle/wheeze, полезно как cough hard negatives | Микрофонные cough-записи, не stethoscope |
| 8 | AudioSet respiratory sounds | Большие слабые метки breathing/cough/sneeze/sniff | Низкая для crackle/wheeze, полезно как noisy pretraining/hard negatives | YouTube-сегменты, слабая разметка, не медицинский stethoscope |

## 1. ICBHI 2017: текущая база

Источник:

- Official challenge: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- Kaggle mirror: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

Что есть:

- `920` WAV-записей длиной примерно от `10` до `90` секунд.
- `126` пациентов.
- `6898` дыхательных циклов.
- Классы цикла: `normal`, `crackle`, `wheeze`, `both`.
- По Kaggle-описанию: `1864` cycles with crackles, `886` with wheezes, `506` with both.

Почему переобучается:

- Небольшое число пациентов важнее числа сегментов. Сегменты одного пациента акустически похожи.
- Класс `both` очень мал.
- Записи сделаны разными устройствами и в разных условиях, но это не заменяет разнообразие пациентов.
- AST-Base имеет около `86M` параметров, поэтому при полном fine-tuning быстро запоминает train-set.

Вывод: ICBHI лучше оставить как основной benchmark и чистый test/validation anchor. Новые данные стоит добавлять в train и, возможно, в отдельный external-validation, но не ломать сравнимость результатов на ICBHI.

## 2. SPRSound

Источник:

- GitHub: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- Статья: `SPRSound: Open-Source SJTU Paediatric Respiratory Sound Database`, IEEE TBioCAS, 2022.

Что есть:

- Pediatric respiratory sound database для возраста от `1` месяца до `18` лет.
- WAV-записи, JSON-аннотации.
- Record-level метки: `Normal`, `CAS`, `DAS`, `CAS & DAS`, `Poor Quality`.
- Event-level метки: `Normal`, `Rhonchi`, `Wheeze`, `Stridor`, `Coarse Crackle`, `Fine Crackle`, `Wheeze+Crackle`.
- В release 2022 указаны `1949` training files и `734` test files; последующие BioCAS-релизы добавляют новые test sets.
- Лицензия репозитория: `CC-BY-4.0`.

Как привести к текущей задаче:

| SPRSound event | Целевой класс проекта |
|---|---|
| `Normal` | `normal` |
| `Wheeze` | `wheeze` |
| `Coarse Crackle`, `Fine Crackle` | `crackle` |
| `Wheeze+Crackle` | `both` |
| `Rhonchi`, `Stridor` | лучше исключить из 4-class обучения или использовать как `other_adventitious` для auxiliary-задачи |
| `Poor Quality` | исключить или использовать только для noise robustness |

Почему это лучший кандидат:

- Есть event-level разметка, поэтому можно обучать тот же fixed-length segment pipeline.
- Формат JSON конвертируется в `index.csv` без ручной разметки.
- Классы crackle/wheeze представлены прямо, а не через диагноз.

Ограничения:

- Датасет детский, а ICBHI содержит разные возрастные группы. Модель может выучить возрастной/устройственный домен.
- Нужно сохранить `dataset=SPRSound` и не перемешивать пациентов между split-ами.
- Для честного дипломного вывода тестировать итоговую модель отдельно на ICBHI и отдельно на held-out SPRSound.

Вердикт: добавлять первым.

## 3. HF_Lung_V1

Источники:

- GitLab: https://gitlab.com/techsupportHF/HF_Lung_V1
- Статья в Scientific Reports / PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8248710/

Что есть:

- Открытая lung sound база для inhalation, exhalation и adventitious sound detection.
- В статье указано `9765` аудиофайлов по `15` секунд.
- Разметка включает `34095` inhalation labels, `18349` exhalation labels.
- Continuous adventitious sounds: `8457` wheeze, `686` stridor, `4740` rhonchus.
- Discontinuous adventitious sounds: `15606` crackles.

Как использовать:

- Для прямой 4-class схемы ICBHI:
  - `wheeze` -> `wheeze`;
  - `crackle` -> `crackle`;
  - участки без CAS/DAS -> `normal`, если они внутри дыхательных фаз;
  - пересечение CAS и DAS -> потенциальный `both`, если аннотации позволяют восстановить overlap.
- Для более надежного обучения:
  - auxiliary-задача `normal / adventitious`;
  - multi-label задача `has_wheeze`, `has_crackle`;
  - предобучение encoder-а, затем fine-tuning на ICBHI 4-class.

Ограничения:

- CAS/DAS не равны напрямую классам ICBHI: `rhonchus` и `stridor` не стоит сливать с `wheeze` без отдельного эксперимента.
- Нужен конвертер интервалов и проверка overlap-разметки.

Вердикт: добавлять вторым, но лучше через multi-label или auxiliary pretraining.

## 4. Fraiwan / KAUH: lung sounds from chest wall

Источник:

- Mendeley Data: https://data.mendeley.com/datasets/jwyy9np4gv/3
- DOI: `10.17632/jwyy9np4gv.3`

Что есть:

- Записи с грудной клетки, полученные электронным стетоскопом `3M Littmann Electronic Stethoscope model 3200`.
- `112` subjects по Kaggle/Mendeley-описаниям.
- Аннотация содержит тип звука: `I` inspiratory, `E` expiratory, `W` wheezes, `C` crackles, `N` normal.
- Есть диагноз специалиста и точка записи на грудной клетке.
- Лицензия: `CC BY 4.0`.

Как использовать:

- `W` -> `wheeze`.
- `C` -> `crackle`.
- `N` -> `normal`.
- Если в аннотациях есть одновременные `W` и `C`, такие сегменты можно привести к `both`; если нет, класс `both` этот набор почти не усилит.

Ограничения:

- Набор небольшой, поэтому он не решит проблему объема сам по себе.
- В файлах есть разные фильтры стетоскопа: `Bell 20-200 Hz`, `Diaphragm 100-500 Hz`, `Extended range 50-500 Hz`. Это может дать сильный domain shift.
- Частоты и формат сегментации нужно нормализовать под текущий `16 kHz mono` pipeline.

Вердикт: полезный небольшой источник для внешней проверки и дообучения, но не главный способ борьбы с переобучением.

## 5. RespiratoryDatabase@TR

Источник:

- Mendeley Data: https://data.mendeley.com/datasets/p9z4h98s6j/1
- DOI: `10.17632/p9z4h98s6j.1`

Что есть:

- `12-channel lung sounds` для каждого пациента.
- Классы тяжести COPD: `COPD0`, `COPD1`, `COPD2`, `COPD3`, `COPD4`.
- Короткие записи не менее `17` секунд.
- Лицензия: `Creative Commons Attribution 4.0 International`.

Как использовать:

- Не подходит напрямую для `normal / crackle / wheeze / both`, если нет event-level crackle/wheeze-разметки.
- Подходит для domain-adaptive pretraining: научить encoder отличать lung-sound domain, позиции и многоканальные записи.
- Можно использовать как external disease-level experiment, если диплом захочет отдельную задачу COPD severity, но это расширяет постановку.

Вердикт: брать после SPRSound/HF_Lung/KAUH, если нужно предобучение на stethoscope-domain.

## 6. Coswara

Источники:

- GitHub: https://github.com/iiscleap/Coswara-Data
- Zenodo: https://zenodo.org/records/7188627
- Статья Scientific Data: https://www.nature.com/articles/s41597-023-02266-0

Что есть:

- Crowdsourced respiratory sounds and speech recordings.
- Каждый участник записывает `9` типов звуков: fast/slow breathing, deep/shallow cough, sustained vowels, counting.
- Есть metadata: возраст, пол, location, health status, comorbidities.
- В GitHub README указана лицензия `CC BY 4.0`.

Как использовать:

- Не смешивать с ICBHI как `normal/crackle/wheeze/both`: это микрофонные mouth/nose recordings, а не auscultation chest-wall recordings.
- Использовать для:
  - self-supervised или supervised pretraining на respiratory audio;
  - hard negatives для отличия дыхания/кашля от stethoscope-events;
  - аугментаций фонового дыхательного шума;
  - проверки, не реагирует ли модель на общий respiratory-sound вместо crackle/wheeze.

Вердикт: полезно для предобучения и регуляризации, но не для прямого supervised-слияния.

## 7. Sound-Dr

Источники:

- GitHub: https://github.com/ReML-AI/Sound-Dr
- arXiv: https://arxiv.org/abs/2201.04581

Что есть:

- Human sounds for respiratory illnesses: coughing, mouth breathing, nose breathing.
- Метаданные, связанные с заболеваниями, включая pneumonia и COVID-19.
- В репозитории есть baseline-код и подготовленные директории `coswara_data`, `coughvid_data`, `sounddr_data`.

Как использовать:

- Только для предобучения или auxiliary disease/noise tasks.
- Не подходит напрямую для crackle/wheeze-cycle classification.

Ограничение:

- Перед загрузкой нужно отдельно проверить лицензию и фактический доступ к данным, потому что GitHub README описывает датасет, но не заменяет формальную лицензию на все аудиофайлы.

Вердикт: вторичный кандидат после stethoscope-наборов.

## 8. CoughVID

Источники:

- EPFL page: https://www.epfl.ch/labs/esl/index-html/datasets/coughviddataset/
- Zenodo: https://zenodo.org/records/4048312
- Scientific Data article: https://www.nature.com/articles/s41597-021-00937-4

Что есть:

- Большой crowdsourced cough corpus для исследования алгоритмов анализа кашля.
- Записи делались через микрофон, обычно до `10` секунд.
- В описании EPFL указано, что тысячи записей дополнительно пересмотрены врачами для разметки медицинских аномалий кашля.
- Публичная часть лежит на Zenodo; private test доступен только через контакт с командой COUGHVID.

Как использовать:

- Только как hard negatives или pretraining для respiratory audio encoder.
- Не использовать как `wheeze`, `crackle` или `normal` для stethoscope-классификации.
- Можно добавить вспомогательную задачу `stethoscope_lung_sound / cough_or_other_respiratory`, чтобы модель меньше путала общую респираторную акустику с хрипами.

Вердикт: полезен для устойчивости, но не решает дефицит размеченных crackle/wheeze-событий.

## 9. AudioSet respiratory sounds

Источник:

- Ontology page: https://research.google.com/audioset/ontology/respiratory_sounds_1.html

Что есть:

- Слабые YouTube-аннотации для respiratory sounds.
- Подкатегории включают `Breathing`, `Cough`, `Sneeze`, `Sniff`.
- На странице ontology указано `834` breathing annotations и `871` cough annotations.

Как использовать:

- Hard negatives и weak pretraining.
- Не использовать как supervised crackle/wheeze labels.
- Можно нарезать фрагменты и обучить encoder отделять respiratory/non-respiratory audio, но качество разметки будет слабым.

Вердикт: полезно только если не хватает предобучения и есть время на фильтрацию.

## 10. Что не стоит делать

Не стоит просто скачать все respiratory/cough datasets и добавить их в один train-set с текущими четырьмя классами. Это создаст ложное увеличение объема:

- `cough` не равен `wheeze`;
- mouth/nose breathing не равно chest auscultation;
- disease label не равен event label;
- rhonchi/stridor нельзя автоматически считать wheeze;
- записи одного источника могут иметь узнаваемые частоты, шумы, микрофоны и preprocessing.

Если смешать источники без контроля, модель может начать классифицировать датасет-источник, а не дыхательный звук.

## 11. Практическая схема интеграции

### Минимальный вариант

1. Добавить `SPRSound` converter:
   - читать WAV + JSON;
   - переводить event intervals из ms в секунды;
   - маппить `Normal/Wheeze/Coarse Crackle/Fine Crackle/Wheeze+Crackle`;
   - исключать `Rhonchi`, `Stridor`, `Poor Quality`;
   - писать общий `index.csv` с колонкой `dataset`.
2. Обучать на `ICBHI train + SPRSound train`.
3. Валидировать отдельно:
   - `ICBHI val`;
   - `SPRSound val`;
   - объединенная validation только как дополнительная метрика.
4. Финальный test оставлять чистым ICBHI, чтобы сравнение с уже полученными экспериментами сохранялось.

### Более надежный вариант

1. Перевести задачу внутри модели в multi-label:
   - `has_crackle`;
   - `has_wheeze`.
2. Восстанавливать 4 класса для отчетных метрик:
   - `0/0 -> normal`;
   - `1/0 -> crackle`;
   - `0/1 -> wheeze`;
   - `1/1 -> both`.
3. Так проще подключить HF_Lung_V1, где есть отдельные CAS/DAS-разметки.
4. Для диплома оставить прежние macro F1 и balanced accuracy по 4 классам.

### Контроль domain shift

В `index.csv` стоит добавить:

```text
dataset, patient_id, recording_id, source_path, start, end, label, sample_rate, device_or_filter
```

Для split:

- не смешивать одного пациента между train/val/test;
- не делать случайный split по сегментам;
- хранить `source_dataset`;
- считать метрики отдельно по источникам;
- показывать в отчете confusion matrix на ICBHI отдельно от mixed validation.

## 12. Как это связано с переобучением

Дополнительные данные помогут, но только часть проблемы. Для AST-Base также стоит рассмотреть:

- заморозить encoder на первые эпохи и обучать только classifier head;
- использовать меньший learning rate для backbone и больший для head;
- усилить SpecAugment: time masking, frequency masking, random time shift;
- добавить early stopping по `validation macro F1`;
- использовать class-balanced sampler без чрезмерного oversampling класса `both`;
- попробовать multi-label loss вместо 4-class CE;
- сравнить с меньшей моделью, например CRNN или ResNet18, на mixed train.

Если цель быстро улучшить текущий дипломный результат, самый прагматичный путь: `SPRSound -> converter -> mixed train -> чистый ICBHI test`. Если есть время на более качественную постановку, лучше сделать multi-label разметку `has_crackle/has_wheeze` и подключить `HF_Lung_V1`.

## 13. Итоговый список источников

- ICBHI 2017 Challenge: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- Kaggle mirror of Respiratory Sound Database: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- SPRSound GitHub: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound
- HF_Lung_V1 GitLab: https://gitlab.com/techsupportHF/HF_Lung_V1
- HF_Lung_V1 article: https://pmc.ncbi.nlm.nih.gov/articles/PMC8248710/
- Fraiwan / KAUH Mendeley dataset: https://data.mendeley.com/datasets/jwyy9np4gv/3
- RespiratoryDatabase@TR Mendeley dataset: https://data.mendeley.com/datasets/p9z4h98s6j/1
- Coswara GitHub: https://github.com/iiscleap/Coswara-Data
- Coswara Zenodo: https://zenodo.org/records/7188627
- Sound-Dr GitHub: https://github.com/ReML-AI/Sound-Dr
- Sound-Dr arXiv: https://arxiv.org/abs/2201.04581
- CoughVID EPFL page: https://www.epfl.ch/labs/esl/index-html/datasets/coughviddataset/
- CoughVID Zenodo: https://zenodo.org/records/4048312
- AudioSet respiratory sounds ontology: https://research.google.com/audioset/ontology/respiratory_sounds_1.html
