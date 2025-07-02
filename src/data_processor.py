import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from IPython.display import display
import boto3
import io
from google.cloud import storage
import sagemaker.huggingface
from transformers import AutoTokenizer
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataProcessor:

    def __init__(self, train_data_path: str, test_data_path: str, model_name: str, bucket_name: str):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.tokenizer = None
        self.dataframe = None
        self.s3 = boto3.client('s3') 
        self.sess = sagemaker.Session()
        self.role = "arn:aws:iam::157226286926:role/clinical_reasoning"
        self.sagemaker_default_bucket = self.sess.default_bucket()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bucket_name = bucket_name


    def __str__(self):
        return f"ClinicalDataProcessor(train_data_path={self.train_data_path}, test_data_path={self.test_data_path}, tokenizer={self.tokenizer}, dataframe={self.dataframe}, s3={self.s3}, sess={self.sess}, role={self.role}, sagemaker_default_bucket={self.sagemaker_default_bucket})"

    def load_data(self, data_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(data_path, sep=',', quoting=1)
            logger.info(f"Loaded {len(df)} clinical cases from {data_path}")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise

    def print_data(self, df: pd.DataFrame):
        display(df)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.dropna(how='all')
            df['input_text'] = df.apply(self._create_input_text, axis=1)
            # Use the expert response as target if available
            if 'Clinician' in df.columns:
                df['output_text'] = df['Clinician']
                df = df[df['output_text'].str.len() > 0]
                logger.info(f"Preprocessed data: {len(df)} valid samples")
                return df[['input_text', 'output_text']]
            else:
                logger.info("No 'Clinician' column found, returning only input_text.")
                return df[['input_text']]
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def _create_input_text(self, row) -> str:
        components = []
        
        # Add county and health level context
        if pd.notna(row.get('County')):
            components.append(f"Location: {row['County']}")
        
        if pd.notna(row.get('Health level')):
            components.append(f"Healthcare Level: {row['Health level']}")
            
        if pd.notna(row.get('Years of Experience')):
            components.append(f"Experience: {row['Years of Experience']} years")
        
        # Add the main clinical prompt
        if pd.notna(row.get('Prompt')):
            components.append(f"Case: {row['Prompt']}")
            
        # Add nursing competency context
        if pd.notna(row.get('Nursing Competency')):
            components.append(f"Nursing Competency: {row['Nursing Competency']}")
            
        # Add clinical panel info
        if pd.notna(row.get('Clinical Panel')):
            components.append(f"Clinical Panel: {row['Clinical Panel']}")
        
        return " | ".join(components)
    
    def _create_labels(self, df: pd.DataFrame) -> List[int]:
        # Simple binary classification based on response length
        # 0: Short response, 1: Comprehensive response
        median_length = df['target_text'].str.len().median()
        return (df['target_text'].str.len() > median_length).astype(int).tolist()
    
    def tokenize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Tokenizing data...")
        try:
            df['input_text'] = df['input_text'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
            if 'output_text' in df.columns:
                df['output_text'] = df['output_text'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))
            return df
        except Exception as e:
            logger.error(f"Error tokenizing data: {str(e)}")
            raise

    def upload_to_gcs(self, df: pd.DataFrame, key: str) -> str:
        try:
            logger.info(f"Uploading processed data to GCloud bucket '{self.bucket_name}' at '{key}'...")
            
            storage_client = storage.Client(project="clinical-reasoning-464409")
            bucket = storage_client.bucket(self.bucket_name)
            blob = bucket.blob(key)
            blob.upload_from_string(df.to_csv(index=False))
            logger.info("Upload to GCloud bucket successful.")
            gcs_uri = f"gs://{self.bucket_name}/{key}"
            return gcs_uri
        
        except Exception as e:
            logger.error(f"Failed to upload to GCloud Storage: {str(e)}")
            raise


def data_processor():
    torch.manual_seed(42)
    np.random.seed(42)

    TRAIN_DATA_PATH = "../data/train.csv"
    TEST_DATA_PATH = "../data/test.csv"
    MODEL_NAME = "t5-small"

    logger.info("Starting clinical reasoning model training pipeline...")

    try:
        processor = ClinicalDataProcessor(TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_NAME, bucket_name='clinican-reasoning-model')

        # Process train data
        logger.info("Processing training data...")
        train_raw = processor.load_data(processor.train_data_path)
        train_processed = processor.preprocess_data(train_raw)
        training_data_uri = processor.upload_to_gcs(train_processed, key="train/train.csv")
        print(f"Train data uploaded to: {training_data_uri}")

        # Process test data
        logger.info("Processing test data...")
        test_raw = processor.load_data(processor.test_data_path)
        test_processed = processor.preprocess_data(test_raw)

        test_data_uri = processor.upload_to_gcs(test_processed, key="test/test.csv")
        print(f"Test data uploaded to: {test_data_uri}")


        return training_data_uri, test_data_uri
        

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    data_processor()
