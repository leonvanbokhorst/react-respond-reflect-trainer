from pathlib import Path
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
import logging
import sys
from typing import Optional
from dotenv import load_dotenv
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] 🤖 %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates the dataset format and displays key information.
    
    Args:
        df: The pandas DataFrame to validate
        
    Raises:
        ValueError: If the dataset doesn't meet the required format
    """
    required_columns = ['text']  # We expect a text column containing conversation turns
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    # Display dataset info
    logger.info("\n📊 Dataset Overview:")
    logger.info(f"Total conversations: {len(df):,}")
    logger.info(f"Columns: {', '.join(df.columns)}")
    
    # Display conversation stats
    logger.info("\n📝 Conversation stats:")
    turns_per_conv = df['text'].apply(len)
    logger.info(f"Average turns per conversation: {turns_per_conv.mean():.1f}")
    logger.info(f"Min turns: {turns_per_conv.min()}")
    logger.info(f"Max turns: {turns_per_conv.max()}")
    
    # Check for empty values
    empty_counts = df.isna().sum()
    if empty_counts.any():
        logger.warning("\n⚠️ Found empty values:")
        for col, count in empty_counts[empty_counts > 0].items():
            logger.warning(f"{col}: {count:,} empty values")
    
    logger.info("\n✅ Dataset validation complete!")

def push_dataset_to_hf(
    local_path: str = "./data",
    repo_id: str = "leonvanbokhorst/react-respond-reflect-v1",
    token: Optional[str] = None
) -> None:
    """
    Pushes a local dataset to Huggingface Hub with style! ✨

    Args:
        local_path (str): Path to the local dataset directory
        repo_id (str): Huggingface repository ID (username/repo-name)
        token (Optional[str]): Huggingface API token. If None, looks for HUGGINGFACE_TOKEN env var
    
    Raises:
        ValueError: If the dataset directory doesn't exist or is empty
        ConnectionError: If there's an issue connecting to Huggingface
    """
    try:
        # Login to Huggingface (if token provided)
        if token:
            login(token)
            logger.info("🔑 Successfully logged into Huggingface!")
        
        # Check if directory exists
        data_path = Path(local_path)
        if not data_path.exists():
            raise ValueError(f"Oops! 😅 The directory {local_path} doesn't exist!")

        # Load all JSON files from the data directory
        json_files = list(data_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"Hey there! 🤔 No JSON files found in {local_path}")

        # Load the JSON data
        conversations = []
        for json_file in json_files:
            logger.info(f"📚 Loading {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both single conversations and arrays of conversations
                if isinstance(data, list):
                    for conv in data:
                        conversations.append({'text': conv})
                else:
                    conversations.append({'text': data})

        # Convert to DataFrame
        df = pd.DataFrame(conversations)
        
        # Validate dataset before pushing
        logger.info("🔍 Validating dataset...")
        validate_dataset(df)
        
        # Convert to Huggingface Dataset
        dataset = Dataset.from_pandas(df)
        
        # Push to hub! 🚀
        logger.info("🚀 Pushing dataset to Huggingface Hub...")
        dataset.push_to_hub(
            repo_id,
            private=False,
            commit_message="Update dataset with new version"
        )
        
        logger.info("✨ Dataset successfully pushed to Huggingface Hub! ✨")
        logger.info(f"🔗 Check it out at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        logger.error(f"❌ Oops! Something went wrong: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the token from environment variables
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("❌ HF_TOKEN not found in .env file!")
        sys.exit(1)
        
    push_dataset_to_hf(token=token)
