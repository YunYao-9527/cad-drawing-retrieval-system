FROM python:3.11-slim

WORKDIR /app

# Install lightweight dependencies (no PyTorch/CLIP/YOLO)
COPY requirements-demo.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY demo_main.py .
COPY create_demo_images.py .
COPY templates/ templates/

# Generate demo images during build
RUN python create_demo_images.py

# Set environment variables
ENV APP_HOST=0.0.0.0
ENV GALLERY_DIR=./demo_gallery

EXPOSE 5000

CMD ["sh", "-c", "python demo_main.py"]
