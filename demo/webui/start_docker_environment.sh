# This command will load a Bash terminal with the same installations as the Docker file
# Useful for debugging the WebUI
docker run -it --rm \
  -p 8000:8000 \
  -v huggingface:/root/.cache/huggingface \
  -v $(pwd)/../..:/app \
  -w /app \
  --gpus all \
  --name deepseek_janus \
  -e MODEL_NAME=deepseek-ai/Janus-1.3B \
  --entrypoint bash \
  julianfl0w/janus:latest
