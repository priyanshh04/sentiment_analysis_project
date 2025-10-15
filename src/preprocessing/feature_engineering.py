"""
Feature Engineering Module
Handles feature extraction and engineering for sentiment analysis
"""

import pandas as pd
import numpy as np
import re
import ast
import logging
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering class for Amazon Reviews sentiment analysis
    """
    
    def __init__(self):
        """Initialize FeatureEngineer"""
        self.feature_names = []
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the feature engineer (learn any necessary parameters)
        
        Args:
            X (pd.DataFrame): Input dataframe
            y: Target variable (ignored)
            
        Returns:
            self: Returns the instance itself
        """
        # No parameters to learn for basic feature engineering
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe by adding engineered features
        
        Args:
            X (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        logger.info("Starting feature engineering...")
        
        # Create a copy to avoid modifying the original
        df = X.copy()
        
        # Create target variables
        df = self._create_target_variables(df)
        
        # Extract helpful features
        df = self._extract_helpful_features(df)
        
        # Create text-based features
        df = self._create_text_features(df)
        
        # Extract temporal features
        df = self._extract_temporal_features(df)
        
        # Create sentiment indicators
        df = self._create_sentiment_indicators(df)
        
        logger.info(f"Feature engineering completed. Added {len(self.feature_names)} new features")
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            X (pd.DataFrame): Input dataframe
            y: Target variable (ignored)
            
        Returns:
            pd.DataFrame: Transformed dataframe
        """
        return self.fit(X, y).transform(X)
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for sentiment analysis"""
        logger.info("Creating target variables...")
        
        def create_binary_sentiment(rating):
            """Binary classification: 1-3 -> 0 (negative), 4-5 -> 1 (positive)"""
            return 0 if rating <= 3 else 1
        
        def create_3class_sentiment(rating):
            """3-class: 1-2 -> 0 (negative), 3 -> 1 (neutral), 4-5 -> 2 (positive)"""
            if rating <= 2:
                return 0  # Negative
            elif rating == 3:
                return 1  # Neutral
            else:
                return 2  # Positive
        
        df['sentiment_binary'] = df['overall'].apply(create_binary_sentiment)
        df['sentiment_3class'] = df['overall'].apply(create_3class_sentiment)
        
        self.feature_names.extend(['sentiment_binary', 'sentiment_3class'])
        
        return df
    
    def _extract_helpful_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from the helpful column"""
        logger.info("Extracting helpful features...")
        
        def parse_helpful_votes(helpful_str):
            """Parse helpful votes from string format"""
            try:
                votes = ast.literal_eval(str(helpful_str))
                if isinstance(votes, list) and len(votes) == 2:
                    helpful_votes = votes[0]
                    total_votes = votes[1]
                    helpfulness_ratio = helpful_votes / total_votes if total_votes > 0 else 0
                    return helpful_votes, total_votes, helpfulness_ratio
                else:
                    return 0, 0, 0
            except:
                return 0, 0, 0
        
        # Apply parsing function
        helpful_data = df['helpful'].apply(parse_helpful_votes)
        
        df['helpful_votes'] = [x[0] for x in helpful_data]
        df['total_votes'] = [x[1] for x in helpful_data]
        df['helpfulness_ratio'] = [x[2] for x in helpful_data]
        
        self.feature_names.extend(['helpful_votes', 'total_votes', 'helpfulness_ratio'])
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        logger.info("Creating text-based features...")
        
        # Handle missing values
        df['reviewText'] = df['reviewText'].fillna('')
        df['summary'] = df['summary'].fillna('')
        
        # Length features
        df['review_length'] = df['reviewText'].apply(len)
        df['review_word_count'] = df['reviewText'].apply(lambda x: len(x.split()))
        df['summary_length'] = df['summary'].apply(len)
        df['summary_word_count'] = df['summary'].apply(lambda x: len(x.split()))
        
        # Average word length
        df['avg_word_length'] = df['reviewText'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        
        # Sentences count
        df['sentence_count'] = df['reviewText'].apply(
            lambda x: len(re.split(r'[.!?]+', x)) - 1 if x else 0
        )
        
        # Words per sentence
        df['words_per_sentence'] = df.apply(
            lambda row: row['review_word_count'] / row['sentence_count'] 
            if row['sentence_count'] > 0 else 0, axis=1
        )
        
        self.feature_names.extend([
            'review_length', 'review_word_count', 'summary_length', 'summary_word_count',
            'avg_word_length', 'sentence_count', 'words_per_sentence'
        ])
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features"""
        logger.info("Extracting temporal features...")
        
        def extract_year(date_string):
            """Extract year from reviewTime"""
            try:
                if pd.notna(date_string):
                    date_obj = datetime.strptime(date_string.strip(), "%m %d, %Y")
                    return date_obj.year
                return None
            except:
                return None
        
        def extract_month(date_string):
            """Extract month from reviewTime"""
            try:
                if pd.notna(date_string):
                    date_obj = datetime.strptime(date_string.strip(), "%m %d, %Y")
                    return date_obj.month
                return None
            except:
                return None
        
        df['review_year'] = df['reviewTime'].apply(extract_year)
        df['review_month'] = df['reviewTime'].apply(extract_month)
        
        # Create seasonal features
        def get_season(month):
            """Get season from month"""
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Fall'
            return 'Unknown'
        
        df['review_season'] = df['review_month'].apply(get_season)
        
        # Days since earliest review
        if df['review_year'].notna().any():
            min_year = df['review_year'].min()
            df['years_since_first'] = df['review_year'] - min_year
        else:
            df['years_since_first'] = 0
        
        self.feature_names.extend(['review_year', 'review_month', 'review_season', 'years_since_first'])
        
        return df
    
    def _create_sentiment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment indicator features"""
        logger.info("Creating sentiment indicators...")
        
        # Punctuation-based features
        df['exclamation_count'] = df['reviewText'].apply(lambda x: x.count('!'))
        df['question_count'] = df['reviewText'].apply(lambda x: x.count('?'))
        df['period_count'] = df['reviewText'].apply(lambda x: x.count('.'))
        
        # Capital letters (emotion intensity)
        df['capital_letter_count'] = df['reviewText'].apply(
            lambda x: sum(1 for c in x if c.isupper())
        )
        df['capital_letter_ratio'] = df.apply(
            lambda row: row['capital_letter_count'] / row['review_length'] 
            if row['review_length'] > 0 else 0, axis=1
        )
        
        # All caps words (strong emotion)
        df['all_caps_words'] = df['reviewText'].apply(
            lambda x: len([word for word in x.split() if word.isupper() and len(word) > 1])
        )
        
        # Positive and negative word patterns (simple heuristic)
        positive_words = ['great', 'excellent', 'amazing', 'perfect', 'love', 'best', 'awesome', 'fantastic']
        negative_words = ['terrible', 'awful', 'worst', 'hate', 'horrible', 'disappointing', 'useless']
        
        df['positive_word_count'] = df['reviewText'].apply(
            lambda x: sum(1 for word in positive_words if word in x.lower())
        )
        df['negative_word_count'] = df['reviewText'].apply(
            lambda x: sum(1 for word in negative_words if word in x.lower())
        )
        
        # Sentiment polarity ratio
        df['sentiment_word_ratio'] = df.apply(
            lambda row: (row['positive_word_count'] - row['negative_word_count']) / 
            (row['positive_word_count'] + row['negative_word_count'] + 1), axis=1
        )
        
        self.feature_names.extend([
            'exclamation_count', 'question_count', 'period_count',
            'capital_letter_count', 'capital_letter_ratio', 'all_caps_words',
            'positive_word_count', 'negative_word_count', 'sentiment_word_ratio'
        ])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names"""
        return self.feature_names.copy()
    
    def get_feature_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the created features"""
        if not self.feature_names:
            return {}
        
        feature_info = {}
        for feature in self.feature_names:
            if feature in df.columns:
                feature_info[feature] = {
                    'type': str(df[feature].dtype),
                    'missing': df[feature].isnull().sum(),
                    'unique_values': df[feature].nunique(),
                    'min': df[feature].min() if pd.api.types.is_numeric_dtype(df[feature]) else None,
                    'max': df[feature].max() if pd.api.types.is_numeric_dtype(df[feature]) else None,
                    'mean': df[feature].mean() if pd.api.types.is_numeric_dtype(df[feature]) else None
                }
        
        return feature_info
