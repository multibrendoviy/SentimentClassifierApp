FROM python:3.8.2-slim-buster

WORKDIR /app/frontend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        gcc

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]