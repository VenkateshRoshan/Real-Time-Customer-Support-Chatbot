import boto3
from pathlib import Path
import tarfile
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_tar():
    model_path = Path("models/customer_support_gpt")  # Path to your model folder
    tar_path = "model.tar.gz"  # Path for the output tar.gz file
    
    # Create a tar.gz file containing all files in the model folder
    with tarfile.open(tar_path, "w:gz") as tar:
        for file_path in model_path.glob("*"):
            if file_path.is_file():
                logger.info(f"Adding {file_path} to tar archive")
                tar.add(file_path, arcname=file_path.name)
    
    return tar_path

def upload_to_s3(tar_path, bucket_name, s3_key):
    # Initialize S3 client
    s3 = boto3.client("s3")
    
    # Upload tar.gz file to S3
    logger.info(f"Uploading {tar_path} to s3://{bucket_name}/{s3_key}")
    s3.upload_file(tar_path, bucket_name, s3_key)
    logger.info("Upload complete!")

# Main code
try:
    bucket_name = 'customer-support-gpt'  # Your S3 bucket name
    s3_key = "models/model.tar.gz"  # S3 key (path in bucket)

    # Create the tar.gz archive
    tar_path = create_model_tar()

    # Upload the tar.gz to S3
    upload_to_s3(tar_path, bucket_name, s3_key)

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    # Clean up the local tar file
    if os.path.exists(tar_path):
        os.remove(tar_path)
        logger.info(f"Deleted local file: {tar_path}")
