# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt-get/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install CPU-only PyTorch and torchvision first to avoid multi-GB CUDA wheels
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and resources
COPY . .

# Grant execution permission to entrypoint script
RUN chmod +x entrypoint.sh

# Expose ports for Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Set container entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command launches Streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
