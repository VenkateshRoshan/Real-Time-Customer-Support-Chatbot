import boto3
from pathlib import Path
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import transformers
import torch
import logging
import tarfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_tar():
    model_path = Path("models/customer_support_gpt")
    tar_path = "model.tar.gz"
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in model_path.glob("*"):
            if file_path.is_file():
                logger.info(f"Adding {file_path} to tar archive")
                tar.add(file_path, arcname=file_path.name)
    
    return tar_path

try:
    # Initialize s3 client
    s3 = boto3.client("s3")
    bucket_name = 'customer-support-gpt'
    
    # Create and upload tar.gz
    tar_path = create_model_tar()
    s3_key = "models/model.tar.gz"  # Changed path
    logger.info(f"Uploading model.tar.gz to s3://{bucket_name}/{s3_key}")
    s3.upload_file(tar_path, bucket_name, s3_key)
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = 'arn:aws:iam::841162707028:role/service-role/AmazonSageMaker-ExecutionRole-20241109T160615'
    
    # Verify IAM role
    iam = boto3.client('iam')
    try:
        iam.get_role(RoleName=role.split('/')[-1])
        logger.info(f"Successfully verified IAM role: {role}")
    except iam.exceptions.NoSuchEntityException:
        logger.error(f"IAM role not found: {role}")
        raise
    
    # Point to the tar.gz file
    model_artifacts = f's3://{bucket_name}/{s3_key}'
    print(f'Model artifacts: {model_artifacts}')
    
    env = {
        "model_path": "/opt/ml/model",
        "max_length": "256",
        "generation_config": '{"max_length":100,"temperature":0.7,"top_p":0.95,"top_k":50,"do_sample":true}'
    }

    try:
        huggingface_model = HuggingFaceModel(
            model_data=model_artifacts,
            role=role,
            transformers_version="4.37.0",  # Explicit version
            pytorch_version="2.1.0",        # Matching your version
            py_version="py310",             # Keep py310
            env=env,
            name="customer-support-gpt"
        )
        
        logger.info("Starting model deployment...")
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.xlarge",
            wait=True
        )
        logger.info("Model deployed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model deployment: {str(e)}")
        raise

except Exception as e:
    logger.error(f"Deployment failed: {str(e)}")
    raise

finally:
    # Clean up the local tar file
    if os.path.exists(tar_path):
        os.remove(tar_path)