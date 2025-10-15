"""
Data Loader Module
Handles loading and basic validation of the Amazon Musical Instruments Reviews dataset
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class for Amazon Musical Instruments Reviews dataset
    """

    def __init__(self, data_path: str):
        """
        Initialize DataLoader

        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None

    def load_raw_data(self, filename: str = "Instruments_Reviews.csv") -> pd.DataFrame:
        """
        Load raw data from CSV file

        Args:
            filename (str): Name of the CSV file

        Returns:
            pd.DataFrame: Raw dataset
        """
        try:
            file_path = self.data_path / "raw" / filename
            logger.info(f"Loading raw data from: {file_path}")

            self.raw_data = pd.read_csv(file_path)
            logger.info(f"Raw data loaded successfully. Shape: {self.raw_data.shape}")

            # Basic validation
            self._validate_raw_data()

            return self.raw_data

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def load_processed_data(self, filename: str = "processed_instruments_reviews.csv") -> pd.DataFrame:
        """
        Load processed data from CSV file

        Args:
            filename (str): Name of the processed CSV file

        Returns:
            pd.DataFrame: Processed dataset
        """
        try:
            file_path = self.data_path / "processed" / filename
            logger.info(f"Loading processed data from: {file_path}")

            self.processed_data = pd.read_csv(file_path)
            logger.info(f"Processed data loaded successfully. Shape: {self.processed_data.shape}")

            return self.processed_data

        except FileNotFoundError:
            logger.warning(f"Processed file not found: {file_path}")
            logger.info("You may need to run the preprocessing pipeline first")
            return None
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise

    def _validate_raw_data(self) -> None:
        """Validate the raw dataset structure"""
        expected_columns = [
            'reviewerID', 'asin', 'reviewerName', 'helpful', 'reviewText', 
            'overall', 'summary', 'unixReviewTime', 'reviewTime'
        ]

        missing_columns = set(expected_columns) - set(self.raw_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for critical missing values
        if self.raw_data['reviewText'].isna().sum() > len(self.raw_data) * 0.1:
            logger.warning("More than 10% of reviews have missing text")

        if self.raw_data['overall'].isna().sum() > 0:
            raise ValueError("Overall ratings contain missing values")

        logger.info("Raw data validation completed successfully")

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset

        Returns:
            Dict[str, Any]: Dataset information
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")

        info = {
            'shape': self.raw_data.shape,
            'columns': self.raw_data.columns.tolist(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'data_types': self.raw_data.dtypes.to_dict(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum(),
            'rating_distribution': self.raw_data['overall'].value_counts().sort_index().to_dict(),
            'date_range': {
                'min_year': self.raw_data['reviewTime'].str.extract(r'(\d{4})$')[0].astype(float).min(),
                'max_year': self.raw_data['reviewTime'].str.extract(r'(\d{4})$')[0].astype(float).max()
            } if 'reviewTime' in self.raw_data.columns else None,
            'unique_reviewers': self.raw_data['reviewerID'].nunique(),
            'unique_products': self.raw_data['asin'].nunique()
        }

        return info

    def save_processed_data(self, data: pd.DataFrame, filename: str = "processed_instruments_reviews.csv") -> None:
        """
        Save processed data to CSV file

        Args:
            data (pd.DataFrame): Processed dataset to save
            filename (str): Name of the file to save
        """
        try:
            file_path = self.data_path / "processed" / filename

            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            data.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to: {file_path}")
            logger.info(f"Saved data shape: {data.shape}")

        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def get_sample_data(self, n_samples: int = 100, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of the data for testing

        Args:
            n_samples (int): Number of samples to return
            random_state (int): Random state for reproducibility

        Returns:
            pd.DataFrame: Sample dataset
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")

        return self.raw_data.sample(n=min(n_samples, len(self.raw_data)), 
                                  random_state=random_state)

# Utility functions for data loading
def load_data(data_path: str, file_type: str = "raw", filename: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load data

    Args:
        data_path (str): Path to data directory
        file_type (str): Type of data to load ('raw' or 'processed')
        filename (str, optional): Specific filename to load

    Returns:
        pd.DataFrame: Loaded dataset
    """
    loader = DataLoader(data_path)

    if file_type == "raw":
        if filename:
            return loader.load_raw_data(filename)
        return loader.load_raw_data()
    elif file_type == "processed":
        if filename:
            return loader.load_processed_data(filename)
        return loader.load_processed_data()
    else:
        raise ValueError("file_type must be 'raw' or 'processed'")

def get_data_summary(data: pd.DataFrame) -> None:
    """
    Print a comprehensive summary of the dataset

    Args:
        data (pd.DataFrame): Dataset to summarize
    """
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Shape: {data.shape}")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumn Information:")
    print("-" * 30)
    for col in data.columns:
        null_count = data[col].isnull().sum()
        null_pct = (null_count / len(data)) * 100
        print(f"{col:20} | {str(data[col].dtype):10} | Nulls: {null_count:4d} ({null_pct:5.1f}%)")

    if 'overall' in data.columns:
        print("\nRating Distribution:")
        print("-" * 30)
        rating_dist = data['overall'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = (count / len(data)) * 100
            print(f"{rating} stars: {count:5d} ({pct:5.1f}%)")
