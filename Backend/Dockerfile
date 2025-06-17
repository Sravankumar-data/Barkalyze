FROM python:3.11

# Install OS-level dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Backend

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


ENV PYTHONPATH="${PYTHONPATH}:/Backend/connect"

CMD ["uvicorn", "connect.api:app", "--host", "0.0.0.0", "--port", "8000"]
