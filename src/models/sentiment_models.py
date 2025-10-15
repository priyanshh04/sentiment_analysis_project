"""
Machine Learning Models Module
Implements various algorithms for sentiment analysis
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. SMOTE will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModelTrainer:
    """
    Main class for training sentiment analysis models
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 test_size: float = 0.2,
                 cv_folds: int = 5,
                 use_smote: bool = True):
        """
        Initialize SentimentModelTrainer
        
        Args:
            random_state (int): Random state for reproducibility
            test_size (float): Test set size ratio
            cv_folds (int): Number of cross-validation folds
            use_smote (bool): Whether to use SMOTE for class imbalance
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_smote = use_smote and IMBALANCED_LEARN_AVAILABLE
        
        # Storage for models and results
        self.models = {}
        self.results = {}
        self.best_model = None
        self.vectorizer = None
        
        # Initialize model configurations
        self._init_model_configs()
    
    def _init_model_configs(self):
        """Initialize model configurations"""
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l2'],
                    'classifier__class_weight': [None, 'balanced']
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__kernel': ['linear', 'rbf'],
                    'classifier__class_weight': [None, 'balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__class_weight': [None, 'balanced']
                }
            },
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'classifier__alpha': [0.1, 1.0, 10.0]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [3, 6],
                    'classifier__learning_rate': [0.1, 0.01]
                }
            }
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    text_column: str = 'reviewText', 
                    target_column: str = 'sentiment_binary',
                    additional_features: Optional[List[str]] = None) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            target_column (str): Name of target column
            additional_features (List[str], optional): Additional numerical features
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training...")
        
        # Remove rows with missing text or target
        df_clean = df.dropna(subset=[text_column, target_column]).copy()
        logger.info(f"Dataset size after cleaning: {len(df_clean)}")
        
        # Prepare features
        X_text = df_clean[text_column]
        y = df_clean[target_column]
        
        # Handle additional numerical features
        X_additional = None
        if additional_features:
            available_features = [feat for feat in additional_features if feat in df_clean.columns]
            if available_features:
                X_additional = df_clean[available_features]
                logger.info(f"Using {len(available_features)} additional features")
        
        # Split the data
        if X_additional is not None:
            X_train_text, X_test_text, X_train_add, X_test_add, y_train, y_test = train_test_split(
                X_text, X_additional, y, test_size=self.test_size, 
                random_state=self.random_state, stratify=y
            )
        else:
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                X_text, y, test_size=self.test_size, 
                random_state=self.random_state, stratify=y
            )
            X_train_add = X_test_add = None
        
        # Store for later use
        self.X_train_text = X_train_text
        self.X_test_text = X_test_text
        self.X_train_additional = X_train_add
        self.X_test_additional = X_test_add
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"Train set size: {len(y_train)}, Test set size: {len(y_test)}")
        logger.info(f"Train set distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test set distribution: {y_test.value_counts().to_dict()}")
        
        return X_train_text, X_test_text, y_train, y_test
    
    def create_pipeline(self, model_name: str) -> Pipeline:
        """
        Create a pipeline for the specified model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Pipeline: Scikit-learn pipeline
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        # Get model
        classifier = self.model_configs[model_name]['model']
        
        # Create pipeline
        if self.use_smote:
            pipeline = ImbPipeline([
                ('tfidf', vectorizer),
                ('smote', SMOTE(random_state=self.random_state)),
                ('classifier', classifier)
            ])
        else:
            pipeline = Pipeline([
                ('tfidf', vectorizer),
                ('classifier', classifier)
            ])
        
        return pipeline
    
    def train_model(self, model_name: str, perform_cv: bool = True) -> Dict[str, Any]:
        """
        Train a single model
        
        Args:
            model_name (str): Name of model to train
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Training {model_name}...")
        
        # Create pipeline
        pipeline = self.create_pipeline(model_name)
        
        # Train the model
        pipeline.fit(self.X_train_text, self.y_train)
        
        # Store the model
        self.models[model_name] = pipeline
        
        # Evaluate the model
        results = self.evaluate_model(model_name, pipeline, perform_cv)
        
        # Store results
        self.results[model_name] = results
        
        logger.info(f"{model_name} training completed")
        return results
    
    def evaluate_model(self, model_name: str, pipeline: Pipeline, perform_cv: bool = True) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model
            pipeline (Pipeline): Trained pipeline
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        results = {'model_name': model_name}
        
        # Predictions
        y_train_pred = pipeline.predict(self.X_train_text)
        y_test_pred = pipeline.predict(self.X_test_text)
        
        # Probability predictions (for AUC)
        try:
            y_test_proba = pipeline.predict_proba(self.X_test_text)[:, 1]
            y_train_proba = pipeline.predict_proba(self.X_train_text)[:, 1]
        except AttributeError:
            y_test_proba = None
            y_train_proba = None
        
        # Classification reports
        train_report = classification_report(self.y_train, y_train_pred, output_dict=True)
        test_report = classification_report(self.y_test, y_test_pred, output_dict=True)
        
        results.update({
            'train_accuracy': train_report['accuracy'],
            'test_accuracy': test_report['accuracy'],
            'train_f1': train_report['macro avg']['f1-score'],
            'test_f1': test_report['macro avg']['f1-score'],
            'train_precision': train_report['macro avg']['precision'],
            'test_precision': test_report['macro avg']['precision'],
            'train_recall': train_report['macro avg']['recall'],
            'test_recall': test_report['macro avg']['recall'],
            'classification_report': test_report,
            'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist()
        })
        
        # ROC AUC if probabilities available
        if y_test_proba is not None:
            try:
                results['train_roc_auc'] = roc_auc_score(self.y_train, y_train_proba)
                results['test_roc_auc'] = roc_auc_score(self.y_test, y_test_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC AUC for {model_name}: {e}")
        
        # Cross-validation
        if perform_cv:
            try:
                cv_scores = cross_val_score(
                    pipeline, self.X_train_text, self.y_train, 
                    cv=self.cv_folds, scoring='f1_macro'
                )
                results.update({
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                })
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {e}")
        
        return results
    
    def train_all_models(self, perform_cv: bool = True) -> Dict[str, Any]:
        """
        Train all configured models
        
        Args:
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            Dict[str, Any]: All training results
        """
        logger.info("Training all models...")
        
        for model_name in self.model_configs.keys():
            try:
                self.train_model(model_name, perform_cv)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Find best model
        self._find_best_model()
        
        logger.info("All models training completed")
        return self.results
    
    def _find_best_model(self) -> None:
        """Find the best performing model based on test F1 score"""
        if not self.results:
            return
        
        best_score = -1
        best_name = None
        
        for model_name, results in self.results.items():
            f1_score = results.get('test_f1', -1)
            if f1_score > best_score:
                best_score = f1_score
                best_name = model_name
        
        if best_name:
            self.best_model = {
                'name': best_name,
                'model': self.models[best_name],
                'score': best_score,
                'results': self.results[best_name]
            }
            logger.info(f"Best model: {best_name} (F1: {best_score:.4f})")
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model results
        
        Returns:
            pd.DataFrame: Results summary
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Train Accuracy': results.get('train_accuracy', 0),
                'Test Accuracy': results.get('test_accuracy', 0),
                'Train F1': results.get('train_f1', 0),
                'Test F1': results.get('test_f1', 0),
                'Test ROC AUC': results.get('test_roc_auc', 0),
                'CV Mean F1': results.get('cv_mean', 0),
                'CV Std F1': results.get('cv_std', 0)
            })
        
        df = pd.DataFrame(summary_data)
        return df.round(4)
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model
        
        Args:
            model_name (str): Name of model to save
            filepath (str): Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'results': self.results.get(model_name, {}),
            'model_name': model_name
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        model_data = joblib.load(filepath)
        return model_data
    
    def predict(self, texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            texts (List[str]): Texts to predict
            model_name (str, optional): Model to use. Uses best model if None.
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model found. Train models first.")
            model = self.best_model['model']
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        return model.predict(texts)
