# Анализ тональности отзывов клиентов онлайн-магазина "Ozon"

Веб-сервис для анализа тональности отзывов клиентов онлайн-магазина "Ozon" (позитив/негатив).

В качестве классификатора тональности используется предобученная модель Bert, дообученная на корпусе
11к отзывов.

## Краткое описание

- Выполняем разведочный анализ данных на отзывах клиентов.
- Дообучаем модель-трансформер Bert на данных клиентов. 
- Выполняем предсказания на данных из файла/с ввода в UI. 
- Получаем данные для предсказания с сайта otzovik.com.

<img src="demo/demo.gif" width="900" height="550" />

___

# Инструкция


## Клонирование

`git clone https://github.com/multibrendoviy/SentimentClassifierApp.git`


Сервис адаптирован под запуск с помощью Docker на локальной машине. Предварительно необходимо установить Docker.

## Docker

- Создание образа Fastapi из директории mlops_sentiment_project

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Создание образа Streamlit из директории mlops_sentiment_project

`docker build -t streamlit:ver1 frontend -f frontend/Dockerfile`

___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

/

`docker compose up -d --build`

---
## Запуск приложения без Docker

Для запуска проекта на локальном хосте без докера предварительно необходимо раскоментировать
следующие строки в файле `config/config.yaml`:

```python
exploratory: 'http://localhost:8000/compute_eda'
train: 'http://localhost:8000/train'
prediction_input: 'http://localhost:8000/predict_from_input'
prediction_from_file: 'http://localhost:8000/predict'
scrape: 'http://localhost:8000/scrape'
 ```
и соответственно закоментировать следующие строки:
```python
#exploratory: 'http://fastapi:8000/compute_eda'
#train: 'http://fastapi:8000/train'
#prediction_input: 'http://fastapi:8000/predict_from_input'
#prediction_from_file: 'http://fastapi:8000/predict'
#scrape: 'http://fastapi:8000/scrape'
 ```

Далее необходимо в папке mlops_sentiment_project создать файл `requirements.txt`, скопировать в него содержимое файлов
`backend/requirments.txt`, `frontend/requirments.txt`. Так же, добавить строку 
`torch==2.2.0+cu118`. В случае отсутсвия GPU с поддержкой CUDA установите `torch==2.2.0`.

После установки пакетов в виртуальное окружение можно запускать приложение.
Сначала необходимо запустить Fastapi. Далее запускаете Streamlit.


## Запуск FastAPI

- Запуск fastapi  

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload --log_config==log_config.yaml`

## Запуск Streamlit

`cd frontend`

`streamlit run main.py`

И приложение будет доступно по адресу http://localhost:8501 

___



