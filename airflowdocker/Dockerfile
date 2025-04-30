# airflowdocker/Dockerfile
FROM apache/airflow:3.0.0

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends default-jdk-headless \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --upgrade pip wheel \
 && pip install --no-cache-dir \
      pyspark \
      scikit-learn \
      pandas \
      numpy \
      matplotlib \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.0/constraints-3.9.txt" \
 && pip install --no-cache-dir mlflow==2.17.2
