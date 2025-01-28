FROM python:3.10-slim

COPY . /Janus

WORKDIR /Janus

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .
RUN pip install --upgrade torch

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MODEL_PATH_APP="models/models-deepseek-ai-Janus-Pro-1B"
ENV MODEL_PATH_APP_JANUSPRO="models/models-deepseek-ai-Janus-Pro-7B"
ENV HF_HOME=/Janus/models
ENV TRANSFORMERS_CACHE=/Janus/models
ENV HF_DATASETS_CACHE=/Janus/models

CMD ["python", "demo/app.py"]
