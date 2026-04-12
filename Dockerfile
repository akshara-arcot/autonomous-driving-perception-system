
from python:3.10-slim
ENV MODEL_TYPE=onnx
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgl1
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python","main.py"] 