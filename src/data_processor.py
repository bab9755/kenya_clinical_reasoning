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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalDataProcessor:

    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.tokenizer = None
        self.dataframe = None
        self.s3 = boto3.client('s3') 
        
    def load_data(self) -> pd.DataFrame:
        try:
            # Read the CSV with proper quoting to handle embedded commas
            df = pd.read_csv(self.data_path, sep=',', quoting=1)  # quoting=1 for QUOTE_ALL
            logger.info(f"Loaded {len(df)} clinical cases from {self.data_path}")
            
            # Display column names for verification
            logger.info(f"Columns: {df.columns.tolist()}")

            self.dataframe = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def print_data(self):
        display(self.dataframe)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:

        try:
            # Remove any completely empty rows
            df = df.dropna(how='all')
            
            # Create a comprehensive input text by combining relevant columns
            df['input_text'] = df.apply(self._create_input_text, axis=1)
            
            # Use the expert response as target
            df['output_text'] = df['Clinician']
            

            # Remove rows with empty target text
            df = df[df['output_text'].str.len() > 0]
            
            logger.info(f"Preprocessed data: {len(df)} valid samples")
            
            return df[['input_text', 'output_text']]
            
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
    
    def upload_to_s3(self, df: pd.DataFrame):
        try:
            logger.info("Uploading processed data to S3 bucket 'clinical_reasoning' at 'train/train.csv'...")
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            self.s3.put_object(
                Bucket="clinical.reasoning",
                Key="train/train.csv",
                Body=csv_buffer.getvalue()
            )
            logger.info("Upload to S3 successful.")
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise


def main():
    """
    Main training pipeline.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    DATA_PATH = "../data/train.csv"
    OUTPUT_DIR = "../models/clinical_reasoning_model"
    MODEL_NAME = "t5-small"
    
    logger.info("Starting clinical reasoning model training pipeline...")
    
    try:
        # Step 1: Data Processing
        logger.info("Step 1: Processing clinical data...")
        processor = ClinicalDataProcessor(DATA_PATH)
        raw_data = processor.load_data()
        processed_data = processor.preprocess_data(raw_data)

        print(processed_data)

        processor.upload_to_s3(processed_data)
        
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
