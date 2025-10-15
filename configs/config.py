"""
Project Configuration File
Sentiment Analysis for Amazon Musical Instruments Reviews
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataConfig:
    """Data-related configuration"""
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    models_path: str = "data/models/"
    
    # File names
    raw_file: str = "Instruments_Reviews.csv"
    processed_file: str = "processed_instruments_reviews.csv"
    
    # Target variable options
    target_binary: str = "sentiment_binary"
    target_3class: str = "sentiment_3class"

@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration"""
    min_text_length: int = 5
    max_text_length: int = 10000
    
    # Text cleaning options
    remove_urls: bool = True
    remove_punctuation: bool = True
    remove_digits: bool = True
    convert_lowercase: bool = True
    remove_stopwords: bool = True
    apply_lemmatization: bool = True
    
    # Feature extraction
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95

@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Class imbalance handling
    use_smote: bool = True
    smote_k_neighbors: int = 5
    
    # Models to train
    models_to_train: List[str] = None
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = [
                'logistic_regression',
                'svm',
                'random_forest',
                'naive_bayes',
                'xgboost'
            ]

@dataclass
class EvaluationConfig:
    """Model evaluation configuration"""
    primary_metric: str = "f1_score"
    metrics_to_calculate: List[str] = None
    
    def __post_init__(self):
        if self.metrics_to_calculate is None:
            self.metrics_to_calculate = [
                'accuracy',
                'precision',
                'recall',
                'f1_score',
                'roc_auc'
            ]

class ProjectConfig:
    """Main project configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        
        # Project paths
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.src_path = os.path.join(self.project_root, "..", "src")
        
    def get_data_path(self, file_type: str = "raw") -> str:
        """Get full path for data files"""
        if file_type == "raw":
            return os.path.join(self.project_root, "..", self.data.raw_data_path)
        elif file_type == "processed":
            return os.path.join(self.project_root, "..", self.data.processed_data_path)
        elif file_type == "models":
            return os.path.join(self.project_root, "..", self.data.models_path)
        else:
            raise ValueError(f"Unknown file_type: {file_type}")

# Create global config instance
config = ProjectConfig()
