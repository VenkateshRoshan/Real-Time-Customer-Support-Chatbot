import boto3
import logging
import sagemaker
from sagemaker.model import Model
import argparse
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def create_model_archive(model_path):
    """
    Create a model archive if needed
    
    Args:
        model_path (str): Path to model files
        
    Returns:
        str: S3 URI of the model archive
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        bucket = 'customer-support-gpt'
        model_key = 'models/model.tar.gz'
        
        # Check if model archive exists in S3
        try:
            s3.head_object(Bucket=bucket, Key=model_key)
            logger.info("Model archive already exists in S3")
        except:
            logger.info("Model archive not found in S3, will be created during deployment")
            
        return f's3://{bucket}/{model_key}'
    except Exception as e:
        logger.error(f"Error creating model archive: {str(e)}")
        raise

def deploy_app(acc_id, region_name, role_arn, ecr_repo_name, endpoint_name="customer-support-chatbot"):
    """
    Deploys a Gradio app as a SageMaker endpoint using an ECR image.
    
    Args:
        acc_id (str): AWS account ID
        region_name (str): AWS region name
        role_arn (str): IAM role ARN for SageMaker
        ecr_repo_name (str): ECR repository name
        endpoint_name (str): SageMaker endpoint name
    """
    try:
        logger.info("Starting SageMaker deployment process...")
        
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        
        # Define the image URI in ECR
        ecr_image = f"{acc_id}.dkr.ecr.{region_name}.amazonaws.com/{ecr_repo_name}:latest"
        logger.info(f"Using ECR image: {ecr_image}")
        
        # Get model archive S3 URI
        model_data = create_model_archive("models/customer_support_gpt")
        
        # Define model configuration
        model_environment = {
            "MODEL_PATH": "/opt/ml/model",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
            "SAGEMAKER_PROGRAM": "inference.py"
        }
        
        # Create model
        logger.info("Creating SageMaker model...")
        model = Model(
            image_uri=ecr_image,
            model_data=model_data,
            role=role_arn,
            sagemaker_session=sagemaker_session,
            env=model_environment,
            enable_network_isolation=False
        )
        
        # Define deployment configuration
        deployment_config = {
            "initial_instance_count": 1,
            "instance_type": "ml.t3.large",
            "endpoint_name": endpoint_name,
            "update_endpoint": True if _endpoint_exists(sagemaker_session, endpoint_name) else False
        }
        
        # Deploy model
        logger.info(f"Deploying model to endpoint: {endpoint_name}")
        logger.info(f"Deployment configuration: {deployment_config}")
        
        predictor = model.deploy(**deployment_config)
        
        logger.info(f"Successfully deployed to endpoint: {endpoint_name}")
        return predictor
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

def _endpoint_exists(sagemaker_session, endpoint_name):
    """Check if SageMaker endpoint already exists"""
    client = sagemaker_session.boto_session.client('sagemaker')
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except client.exceptions.ClientError:
        return False

def main():
    parser = argparse.ArgumentParser(description="Deploy Gradio app to SageMaker")
    parser.add_argument("--account_id", type=str, required=True,
                      help="AWS Account ID")
    parser.add_argument("--region", type=str, required=True,
                      help="AWS Region")
    parser.add_argument("--role_arn", type=str, required=True,
                      help="IAM Role ARN for SageMaker")
    parser.add_argument("--ecr_repo_name", type=str, required=True,
                      help="ECR Repository name")
    parser.add_argument("--endpoint_name", type=str,
                      default="customer-support-chatbot",
                      help="SageMaker Endpoint Name")
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting deployment process...")
        deploy_app(
            args.account_id,
            args.region,
            args.role_arn,
            args.ecr_repo_name,
            args.endpoint_name
        )
        logger.info("Deployment completed successfully!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()