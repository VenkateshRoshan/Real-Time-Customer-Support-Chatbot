import boto3
import logging
import sagemaker
from sagemaker.model import Model
import argparse
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_app(acc_id, region_name, role_arn, ecr_repo_name, endpoint_name="customer-support-chatbot"):
    """
    Deploys a Gradio app as a SageMaker endpoint using an ECR image.
    
    Args:
        acc_id (str): AWS account ID
        region_name (str): AWS region name
        role_arn (str): IAM role ARN for SageMaker
        ecr_repo_name (str): ECR repository name
        endpoint_name (str): SageMaker endpoint name (default: "customer-support-chatbot")
    """
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Define the image URI in ECR
    ecr_image = f"{acc_id}.dkr.ecr.{region_name}.amazonaws.com/{ecr_repo_name}:latest"

    # Define model
    model = Model(
        image_uri=ecr_image,
        role=role_arn,
        sagemaker_session=sagemaker_session,
        entry_point="serve",
    )

    # Deploy model as a SageMaker endpoint
    logger.info(f"Starting deployment of Gradio app to SageMaker endpoint {endpoint_name}...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.t3.large", #"ml.g4dn.xlarge",
        endpoint_name=endpoint_name
    )
    logger.info(f"Gradio app deployed successfully to endpoint: {endpoint_name}")

if __name__ == "__main__":
    # Parse arguments from CLI
    parser = argparse.ArgumentParser(description="Deploy Gradio app to SageMaker")
    parser.add_argument("--account_id", type=str, required=True, help="AWS Account ID")
    parser.add_argument("--region", type=str, required=True, help="AWS Region")
    parser.add_argument("--role_arn", type=str, required=True, help="IAM Role ARN for SageMaker")
    parser.add_argument("--ecr_repo_name", type=str, required=True, help="ECR Repository name")
    parser.add_argument("--endpoint_name", type=str, default="customer-support-chatbot", help="SageMaker Endpoint Name")
    args = parser.parse_args()

    # Deploy the Gradio app to SageMaker
    deploy_app(args.account_id, args.region, args.role_arn, args.ecr_repo_name, args.endpoint_name)
