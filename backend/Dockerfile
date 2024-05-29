FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime


WORKDIR /app/backend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++


RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt


EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", \
    "8000" , "--reload", "--log-config", "log_config.yaml"]