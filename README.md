# Photo Search by Face

Веб-приложение для индексации фотографий, обнаружения лиц, кластеризации по личности и поиска по лицу.

Стек: Python, Flask, dlib/face_recognition, scikit-learn, SQLite, HTMX.

## Возможности

- Загрузка фото в проекты
- Автоматическое обнаружение лиц и извлечение эмбеддингов (модель CNN или HOG)
- Кластеризация лиц (DBSCAN) для группировки по личности
- Поиск похожих фото по загружаемому изображению
- Присвоение имён (персон) кластерам и отдельным лицам
- UI на HTMX без JavaScript-фреймворков

## Требования

- Python 3.10 или 3.11
- Windows 10/11 (Linux/macOS работают с минимальными правками)
- Интернет для первоначальной установки

## Быстрый старт (Windows)

### 1. Установить Python

Скачать и установить **Python 3.10 или 3.11** с https://www.python.org/downloads/

> При установке отметить галочку **"Add Python to PATH"**.

### 2. Запустить setup

Дважды кликнуть `setup.bat`. Он создаст виртуальное окружение и установит все зависимости,
включая `dlib` (pre-built wheel — компилятор C++ не нужен).

### 3. Запустить приложение

Дважды кликнуть `run.bat`, затем открыть http://127.0.0.1:5000

## Ручная установка (любая ОС)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
flask --app app.web run
```

## Конфигурация

| Переменная окружения | По умолчанию | Описание |
|----------------------|--------------|----------|
| `APP_DATA_DIR` | `./data` | Директория для БД и загруженных фото |

Пример:
```bat
set APP_DATA_DIR=D:\my_photos_data
run.bat
```

## Структура данных

```
data/
  face_index.db                        ← SQLite база данных
  projects/<project_id>/
    photos/                            ← оригинальные загрузки
    thumbs/                            ← миниатюры фото (200px)
    faces/                             ← вырезанные лица (100px)
```

## CLI

```bash
python -m app.main project create --name "Мой проект"
python -m app.main project list
python -m app.main photos add --project <id> --dir ./photos
python -m app.main index --project <id>
python -m app.main cluster --project <id> [--eps 0.6] [--min-samples 2]
python -m app.main search --project <id> --query-image ./query.jpg
python -m app.main person list --project <id>
```

## Развёртывание без интернета

На машине с интернетом скачать пакеты:
```bat
pip download -r requirements.txt -d packages\
```
Скопировать папку `packages\` вместе с проектом. В `setup.bat` заменить строку установки на:
```bat
pip install --no-index --find-links=packages -r requirements.txt
```

## Модель обнаружения лиц

В настройках каждого проекта можно выбрать модель:

- **CNN** (по умолчанию) — точная, находит лица под углом и частично скрытые, медленнее на CPU
- **HOG** — быстрая, лучше работает для фронтальных лиц

## Порог похожести при поиске

- Меньше значение `threshold` → строже совпадения (меньше ложных срабатываний)
- Больше значение `threshold` → мягче совпадения (выше полнота)
- Типичное стартовое значение: `0.6`

## Параметры кластеризации

- **eps** — максимальное расстояние между лицами в эмбеддинг-пространстве для их объединения в кластер. Рекомендуемый диапазон: `0.5–0.65`
- **min_samples** — минимальное количество лиц для формирования кластера (2 или больше)
