# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git

# Install pip packages without caching
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy .env file to the working directory
COPY .env /app/.env

# Set environment variables from .env file
ENV $(cat /app/.env | xargs)

# Expose port 7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
