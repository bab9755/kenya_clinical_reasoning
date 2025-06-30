#!/usr/bin/env python3
"""
AWS Deployment Script for Clinical Reasoning Model
This script handles the complete deployment pipeline:
1. Upload model artifacts to S3
2. Create SageMaker model
3. Create endpoint configuration
4. Deploy endpoint
5. Test inference
"""

import os
import json
import boto3
import tarfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalModelDeployment:
    def __init__(self, 
                 aws_region: str = 'us-east-1',
                 s3_bucket: str = None,
                 model_name: str = 'clinical-reasoning-model',
                 endpoint_name: str = None):
        """
        Initialize the deployment configuration
        
        Args:
            aws_region: AWS region for deployment
            s3_bucket: S3 bucket name (will be created if doesn't exist)
            model_name: Name for the SageMaker model
            endpoint_name: Name for the SageMaker endpoint
        """
        self.aws_region = aws_region
        self.model_name = model_name
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        # Generate names with timestamp
        self.s3_bucket = s3_bucket or f'clinical-model-{self.timestamp}'
        self.endpoint_name = endpoint_name or f'clinical-endpoint-{self.timestamp}'
        self.endpoint_config_name = f'clinical-config-{self.timestamp}'
        
        # Initialize AWS clients
        self.session = boto3.Session(region_name=self.aws_region)
        self.s3_client = self.session.client('s3')
        self.sagemaker_client = self.session.client('sagemaker')
        self.iam_client = self.session.client('iam')
        
        # Get account ID and execution role
        self.account_id = self.session.client('sts').get_caller_identity()['Account']
        self.execution_role = self._get_or_create_execution_role()
        
        logger.info(f"Initialized deployment for model: {self.model_name}")
        logger.info(f"S3 Bucket: {self.s3_bucket}")
        logger.info(f"Endpoint: {self.endpoint_name}")

    def _get_or_create_execution_role(self) -> str:
        """Get or create SageMaker execution role"""
        role_name = 'SageMakerExecutionRole-ClinicalModel'
        role_arn = f'arn:aws:iam::{self.account_id}:role/{role_name}'
        
        try:
            # Check if role exists
            self.iam_client.get_role(RoleName=role_name)
            logger.info(f"Using existing IAM role: {role_arn}")
            return role_arn
        except self.iam_client.exceptions.NoSuchEntityException:
            logger.info(f"Creating new IAM role: {role_name}")
            
            # Create role with trust policy
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for Clinical Reasoning Model SageMaker endpoint'
            )
            
            # Attach necessary policies
            policies = [
                'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess'
            ]
            
            for policy_arn in policies:
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
            
            # Wait for role to be available
            time.sleep(30)
            logger.info(f"Created IAM role: {role_arn}")
            return role_arn

    def create_model_artifacts(self, model_dir: str = './fine_tuned_model') -> str:
        """
        Create model.tar.gz with model artifacts and inference code
        
        Args:
            model_dir: Directory containing the fine-tuned model
            
        Returns:
            Path to the created tar.gz file
        """
        logger.info("Creating model artifacts...")
        
        # Create inference script
        inference_script = '''
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def model_fn(model_dir):
    """Load the model and tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, content_type):
    """Parse input data"""
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Extract text from input
    text = input_data.get("text", "")
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Return predictions
    predicted_class = torch.argmax(predictions, dim=-1).item()
    confidence = torch.max(predictions).item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "all_probabilities": predictions.tolist()[0]
    }

def output_fn(prediction, accept):
    """Format output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
        
        # Write inference script to model directory
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'inference.py'), 'w') as f:
            f.write(inference_script)
        
        # Create requirements.txt for the model
        model_requirements = '''
torch==2.0.1
transformers==4.30.0
numpy
'''
        with open(os.path.join(model_dir, 'requirements.txt'), 'w') as f:
            f.write(model_requirements)
        
        # Create tar.gz file
        tar_path = 'model.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(model_dir, arcname='.')
        
        logger.info(f"Model artifacts created: {tar_path}")
        return tar_path

    def upload_to_s3(self, tar_path: str) -> str:
        """
        Upload model artifacts to S3
        
        Args:
            tar_path: Path to the model.tar.gz file
            
        Returns:
            S3 URI of the uploaded model
        """
        logger.info(f"Uploading model to S3 bucket: {self.s3_bucket}")
        
        # Create bucket if it doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
        except:
            if self.aws_region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.s3_bucket)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.s3_bucket,
                    CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                )
            logger.info(f"Created S3 bucket: {self.s3_bucket}")
        
        # Upload model
        s3_key = f'models/{self.model_name}/model.tar.gz'
        self.s3_client.upload_file(tar_path, self.s3_bucket, s3_key)
        
        s3_uri = f's3://{self.s3_bucket}/{s3_key}'
        logger.info(f"Model uploaded to: {s3_uri}")
        return s3_uri

    def create_sagemaker_model(self, model_s3_uri: str) -> str:
        """
        Create SageMaker model
        
        Args:
            model_s3_uri: S3 URI of the model artifacts
            
        Returns:
            Model ARN
        """
        logger.info(f"Creating SageMaker model: {self.model_name}")
        
        # Get the appropriate container image for PyTorch
        container_image = f'763104351884.dkr.ecr.{self.aws_region}.amazonaws.com/pytorch-inference:2.0.1-transformers4.28.1-cpu-py310-ubuntu20.04-sagemaker'
        
        create_model_response = self.sagemaker_client.create_model(
            ModelName=self.model_name,
            PrimaryContainer={
                'Image': container_image,
                'ModelDataUrl': model_s3_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': self.aws_region
                }
            },
            ExecutionRoleArn=self.execution_role
        )
        
        model_arn = create_model_response['ModelArn']
        logger.info(f"Created SageMaker model: {model_arn}")
        return model_arn

    def create_endpoint_config(self, instance_type: str = 'ml.t2.medium') -> str:
        """
        Create endpoint configuration
        
        Args:
            instance_type: EC2 instance type for the endpoint
            
        Returns:
            Endpoint configuration ARN
        """
        logger.info(f"Creating endpoint configuration: {self.endpoint_config_name}")
        
        create_config_response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=self.endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'primary',
                    'ModelName': self.model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1
                }
            ]
        )
        
        config_arn = create_config_response['EndpointConfigArn']
        logger.info(f"Created endpoint configuration: {config_arn}")
        return config_arn

    def create_endpoint(self) -> str:
        """
        Create and deploy the endpoint
        
        Returns:
            Endpoint ARN
        """
        logger.info(f"Creating endpoint: {self.endpoint_name}")
        
        create_endpoint_response = self.sagemaker_client.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name
        )
        
        endpoint_arn = create_endpoint_response['EndpointArn']
        logger.info(f"Endpoint creation initiated: {endpoint_arn}")
        
        # Wait for endpoint to be in service
        logger.info("Waiting for endpoint to be in service...")
        waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(
            EndpointName=self.endpoint_name,
            WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
        )
        
        logger.info(f"Endpoint is now in service: {self.endpoint_name}")
        return endpoint_arn

    def test_endpoint(self, test_text: str = None) -> Dict[str, Any]:
        """
        Test the deployed endpoint
        
        Args:
            test_text: Text to test the endpoint with
            
        Returns:
            Prediction results
        """
        if test_text is None:
            test_text = "A 65-year-old patient presents with chest pain and shortness of breath."
        
        logger.info("Testing endpoint...")
        
        # Create runtime client for inference
        runtime_client = self.session.client('sagemaker-runtime')
        
        # Prepare input data
        input_data = {"text": test_text}
        
        # Make prediction
        response = runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Test prediction result: {result}")
        return result

    def get_deployment_info(self) -> Dict[str, str]:
        """Get deployment information"""
        return {
            'model_name': self.model_name,
            'endpoint_name': self.endpoint_name,
            'endpoint_config_name': self.endpoint_config_name,
            's3_bucket': self.s3_bucket,
            'aws_region': self.aws_region,
            'execution_role': self.execution_role,
            'timestamp': self.timestamp
        }

    def cleanup(self):
        """Clean up AWS resources"""
        logger.info("Cleaning up AWS resources...")
        
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=self.endpoint_name)
            logger.info(f"Deleted endpoint: {self.endpoint_name}")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint: {e}")
        
        try:
            # Delete endpoint configuration
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=self.endpoint_config_name)
            logger.info(f"Deleted endpoint config: {self.endpoint_config_name}")
        except Exception as e:
            logger.warning(f"Failed to delete endpoint config: {e}")
        
        try:
            # Delete model
            self.sagemaker_client.delete_model(ModelName=self.model_name)
            logger.info(f"Deleted model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to delete model: {e}")

def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Clinical Reasoning Model to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--bucket', help='S3 bucket name (optional)')
    parser.add_argument('--model-dir', default='./fine_tuned_model', help='Model directory')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='Instance type')
    parser.add_argument('--test-text', help='Text to test the endpoint with')
    parser.add_argument('--cleanup', action='store_true', help='Clean up resources after testing')
    
    args = parser.parse_args()
    
    try:
        # Initialize deployment
        deployment = ClinicalModelDeployment(
            aws_region=args.region,
            s3_bucket=args.bucket
        )
        
        # Step 1: Create model artifacts
        tar_path = deployment.create_model_artifacts(args.model_dir)
        
        # Step 2: Upload to S3
        model_s3_uri = deployment.upload_to_s3(tar_path)
        
        # Step 3: Create SageMaker model
        model_arn = deployment.create_sagemaker_model(model_s3_uri)
        
        # Step 4: Create endpoint configuration
        config_arn = deployment.create_endpoint_config(args.instance_type)
        
        # Step 5: Create and deploy endpoint
        endpoint_arn = deployment.create_endpoint()
        
        # Step 6: Test endpoint
        test_result = deployment.test_endpoint(args.test_text)
        
        # Print deployment information
        print("\n" + "="*50)
        print("DEPLOYMENT SUCCESSFUL!")
        print("="*50)
        
        info = deployment.get_deployment_info()
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print(f"\nTest Result: {test_result}")
        
        print("\n" + "="*50)
        print("To make predictions, use:")
        print(f"aws sagemaker-runtime invoke-endpoint --endpoint-name {deployment.endpoint_name} --content-type 'application/json' --body '{{\"text\": \"your clinical text here\"}}' output.json")
        print("="*50)
        
        # Cleanup if requested
        if args.cleanup:
            input("\nPress Enter to clean up resources...")
            deployment.cleanup()
        
        # Clean up local tar file
        if os.path.exists(tar_path):
            os.remove(tar_path)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()
