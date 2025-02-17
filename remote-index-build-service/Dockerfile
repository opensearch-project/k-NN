FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
ENV PYTHONPATH=/code/app

ENV REQUEST_STORE_MAX_SIZE=1000
ENV REQUEST_STORE_TTL_SECONDS=900
ENV REQUEST_STORE_TYPE='memory'

ENV GPU_MEMORY_LIMIT=3.0
ENV CPU_MEMORY_LIMIT=32.0

ENV MAX_WORKERS=5

ENV SERVICE_NAME='remote-vector-index-builder'

ENV LOG_LEVEL="INFO"

CMD ["fastapi", "run", "app/main.py", "--port", "80"]