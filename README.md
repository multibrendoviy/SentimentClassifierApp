# Инструкция


## Запуск FastAPI

- Запуск fastapi  

`cd backend`

`uvicorn main:app --host=0.0.0.0 --port=8000 --reload --log_config==log_config.yaml`

## Запуск Streamlit

`cd frontend`

`streamlit run main.py`

И приложение будет доступно по адресу http://localhost:8501 

___


## Docker

- Запуск образа Fastapi из директории mlops_sentiment_project

`docker build -t fastapi:ver1 backend -f backend/Dockerfile`

- Запуск образа Streamlit из директории mlops_sentiment_project

`docker build -t streamlit:ver1 frontend -f frontend/Dockerfile`

___

## Docker Compose

- Сборка сервисов из образов внутри backend/frontend и запуск контейнеров в автономном режиме

`docker compose up -d`

/

`docker compose up -d --build`


