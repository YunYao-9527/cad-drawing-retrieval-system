FROM python:3.11-slim

WORKDIR /app

# Install lightweight dependencies (no PyTorch/CLIP/YOLO)
COPY requirements-demo.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (only what demo needs)
COPY demo_main.py .
COPY templates/ templates/
COPY demo_gallery/ demo_gallery/

# Set environment variables
ENV APP_HOST=0.0.0.0
ENV GALLERY_DIR=./demo_gallery

EXPOSE 5000

CMD ["python", "demo_main.py"]
