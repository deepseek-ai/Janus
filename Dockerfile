# Use the PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory into the container
COPY . /app

# Install necessary Python packages
RUN pip install -e . && \
    pip install fastapi python-multipart uvicorn

# Set the entrypoint for the container to launch your FastAPI app
CMD ["python", "demo/fastapi_app.py"]

