from pathlib import Path
import os
from typing import Dict, Any

class Config:
    # Project structure
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    MODEL_PATH = MODELS_DIR / "customer_support_gpt"
    
    # Model configurations
    MODEL_NAME = "EleutherAI/gpt-neo-125M"
    MAX_LENGTH = 256
    
    # Training configurations
    TRAIN_CONFIG: Dict[str, Any] = {
        "batch_size": 4,
        "learning_rate": 2e-5,
        "epochs": 3,
        "weight_decay": 0.01,
        "max_length": MAX_LENGTH,
    }
    
    # Generation configurations
    GENERATION_CONFIG: Dict[str, Any] = {
        "max_length": 100,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True
    }
    
    # Gradio configurations
    GRADIO_CONFIG: Dict[str, Any] = {
        "title": "Customer Support Chatbot",
        "description": "Ask your questions to the customer support bot!",
        "examples": [
            "How do I reset my password?",
            "What are your shipping policies?",
            "I want to return a product."
        ],
        "share": False
    }
    
    # MLflow configurations
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    EXPERIMENT_NAME = "customer-support-chatbot"
    
    # AWS/SageMaker configurations
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET = os.getenv("S3_BUCKET", "customer-support-chatbot")
    SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE")
    
    # DVC configurations
    DVC_REMOTE_NAME = "s3-storage"
    DVC_REMOTE_URL = f"s3://{S3_BUCKET}/dvc"