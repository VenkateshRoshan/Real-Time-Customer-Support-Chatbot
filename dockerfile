# # Use Python 3.10 slim as base image
FROM python:3.10-slim

# # Set the working directory
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install dependencies
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     git

# # Install pip packages without caching
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# RUN pip install --no-cache-dir -r requirements.txt

# # # Copy .env file to the working directory
# # COPY .env /app/.env

# # # Set environment variables from .env file
# # ENV $(cat /app/.env | xargs)

# # Expose port 7860
# EXPOSE 7860



# # Run the application
# CMD ["python", "app.py"]

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/conda/bin:${PATH}"

# Get args for AWS credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# get args as env variables
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# Install system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libomp-dev \
    libopenblas-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /opt/ml/code

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and serve script
COPY app.py .
COPY serve.sh .

# Make the serve script executable
RUN chmod +x serve.sh

# Expose the Gradio port
EXPOSE 8080

# Set entry point to the serve script
ENTRYPOINT ["./serve.sh"]