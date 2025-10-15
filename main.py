"""
Main Training Script
Complete pipeline for sentiment analysis on Amazon Musical Instruments Reviews
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import custom modules
try:
    from utils.data_loader import DataLoader, get_data_summary
    from preprocessing.feature_engineering import FeatureEngineer
    from preprocessing.text_preprocessing import TextPreprocessor
    from models.sentiment_models import SentimentModelTrainer
    from evaluation.model_evaluation import ModelEvaluator, quick_evaluate_models
    from configs.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the correct directories")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    """
    Complete sentiment analysis pipeline
    """

    def __init__(self, data_path: str = "data"):
        """
        Initialize the pipeline

        Args:
            data_path (str): Path to data directory
        """
        self.data_path = Path(data_path)
        self.results = {}
        self.models = {}
        self.processed_data = None

        # Create directories
        self.data_path.mkdir(exist_ok=True)
        (self.data_path / "raw").mkdir(exist_ok=True)
        (self.data_path / "processed").mkdir(exist_ok=True)
        (self.data_path / "models").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

    def load_data(self, filename: str = "Instruments_Reviews.csv") -> pd.DataFrame:
        """Load and validate the dataset"""
        logger.info("Loading dataset...")

        loader = DataLoader(self.data_path)

        try:
            # Try to load processed data first
            processed_data = loader.load_processed_data()
            if processed_data is not None:
                logger.info("Loaded existing processed data")
                self.processed_data = processed_data
                return processed_data
        except:
            pass

        # Load raw data
        raw_data = loader.load_raw_data(filename)

        # Print data summary
        print("\nDATA SUMMARY:")
        print("=" * 50)
        get_data_summary(raw_data)

        return raw_data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and text preprocessing"""
        logger.info("Starting data preprocessing...")

        # Step 1: Feature Engineering
        logger.info("Step 1: Feature Engineering")
        feature_engineer = FeatureEngineer()
        data_with_features = feature_engineer.fit_transform(data)

        # Print feature engineering results
        feature_info = feature_engineer.get_feature_info(data_with_features)
        logger.info(f"Added {len(feature_engineer.get_feature_names())} new features")

        # Step 2: Text Preprocessing
        logger.info("Step 2: Text Preprocessing")
        text_preprocessor = TextPreprocessor(
            lowercase=config.preprocessing.convert_lowercase,
            remove_punctuation=config.preprocessing.remove_punctuation,
            remove_digits=config.preprocessing.remove_digits,
            remove_stopwords=config.preprocessing.remove_stopwords,
            apply_lemmatization=config.preprocessing.apply_lemmatization
        )

        # Preprocess review text
        data_with_features['reviewText_processed'] = text_preprocessor.fit_transform(
            data_with_features['reviewText'].fillna('')
        )

        # Preprocess summary text
        data_with_features['summary_processed'] = text_preprocessor.fit_transform(
            data_with_features['summary'].fillna('')
        )

        # Save processed data
        self.save_processed_data(data_with_features)

        logger.info("Data preprocessing completed")
        self.processed_data = data_with_features
        return data_with_features

    def train_models(self, data: pd.DataFrame) -> dict:
        """Train all sentiment analysis models"""
        logger.info("Starting model training...")

        # Initialize trainer
        trainer = SentimentModelTrainer(
            random_state=config.model.random_state,
            test_size=config.model.test_size,
            cv_folds=config.model.cv_folds,
            use_smote=config.model.use_smote
        )

        # Additional features to include
        additional_features = [
            'helpful_votes', 'total_votes', 'helpfulness_ratio',
            'review_length', 'review_word_count', 'summary_length',
            'exclamation_count', 'question_count', 'capital_letter_ratio',
            'positive_word_count', 'negative_word_count'
        ]

        # Use processed text
        text_column = 'reviewText_processed'
        target_column = config.data.target_binary

        # Prepare data
        trainer.prepare_data(data, text_column, target_column, additional_features)

        # Train all models
        results = trainer.train_all_models(perform_cv=True)

        # Store results and models
        self.results = results
        self.models = trainer.models
        self.trainer = trainer

        # Print results summary
        summary_df = trainer.get_results_summary()
        print("\nMODEL TRAINING RESULTS:")
        print("=" * 60)
        print(summary_df)

        logger.info("Model training completed")
        return results

    def evaluate_models(self) -> None:
        """Comprehensive model evaluation"""
        logger.info("Starting model evaluation...")

        if not self.results:
            logger.error("No models to evaluate. Train models first.")
            return

        # Create evaluator
        evaluator = ModelEvaluator(save_plots=True, plot_dir="plots")

        # Prepare evaluation data
        evaluation_results = {}

        for model_name, model in self.models.items():
            try:
                # Get predictions
                y_test_pred = model.predict(self.trainer.X_test_text)

                # Get probabilities if available
                try:
                    y_test_proba = model.predict_proba(self.trainer.X_test_text)[:, 1]
                except:
                    y_test_proba = None

                # Evaluate
                eval_results = evaluator.evaluate_single_model(
                    self.trainer.y_test, 
                    y_test_pred, 
                    y_test_proba, 
                    model_name
                )

                evaluation_results[model_name] = eval_results

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue

        # Generate comparison and plots
        if evaluation_results:
            comparison_df = evaluator.compare_models(evaluation_results)
            print("\nMODEL EVALUATION COMPARISON:")
            print("=" * 50)
            print(comparison_df)

            # Generate plots
            evaluator.plot_metrics_comparison(comparison_df)

            # Generate report
            evaluator.save_evaluation_report(evaluation_results, "evaluation_report.txt")

        logger.info("Model evaluation completed")

    def save_processed_data(self, data: pd.DataFrame) -> None:
        """Save processed dataset"""
        filepath = self.data_path / "processed" / config.data.processed_file
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

    def save_models(self) -> None:
        """Save all trained models"""
        if not self.models:
            logger.warning("No models to save")
            return

        models_dir = self.data_path / "models"

        for model_name, model in self.models.items():
            try:
                filepath = models_dir / f"{model_name}_model.joblib"
                self.trainer.save_model(model_name, str(filepath))
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")

        # Save best model separately
        if hasattr(self.trainer, 'best_model') and self.trainer.best_model:
            best_model_path = models_dir / "best_model.joblib"
            best_model_name = self.trainer.best_model['name']
            self.trainer.save_model(best_model_name, str(best_model_path))
            logger.info(f"Best model ({best_model_name}) saved to {best_model_path}")

    def save_results(self) -> None:
        """Save training results to JSON"""
        if not self.results:
            logger.warning("No results to save")
            return

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[model_name][key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_results[model_name][key] = int(value)
                elif isinstance(value, np.floating):
                    json_results[model_name][key] = float(value)
                else:
                    json_results[model_name][key] = value

        # Add metadata
        json_results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(self.processed_data) if self.processed_data is not None else 0,
            'best_model': self.trainer.best_model['name'] if hasattr(self.trainer, 'best_model') and self.trainer.best_model else None
        }

        filepath = Path("reports") / "training_results.json"
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def run_complete_pipeline(self, filename: str = "Instruments_Reviews.csv") -> None:
        """Run the complete sentiment analysis pipeline"""
        logger.info("Starting complete sentiment analysis pipeline...")

        try:
            # Step 1: Load data
            raw_data = self.load_data(filename)

            # Step 2: Preprocess data
            processed_data = self.preprocess_data(raw_data)

            # Step 3: Train models
            self.train_models(processed_data)

            # Step 4: Evaluate models
            self.evaluate_models()

            # Step 5: Save everything
            self.save_models()
            self.save_results()

            logger.info("Complete pipeline finished successfully!")

            # Print final summary
            print("\n" + "="*60)
            print("SENTIMENT ANALYSIS PIPELINE COMPLETED!")
            print("="*60)
            print(f"Dataset processed: {len(processed_data)} reviews")
            print(f"Models trained: {len(self.models)}")
            if hasattr(self.trainer, 'best_model') and self.trainer.best_model:
                best_name = self.trainer.best_model['name']
                best_score = self.trainer.best_model['score']
                print(f"Best model: {best_name} (F1: {best_score:.4f})")
            print("\nFiles generated:")
            print("- Processed data: data/processed/processed_instruments_reviews.csv")
            print("- Models: data/models/")
            print("- Plots: plots/")
            print("- Reports: reports/")
            print("\nPipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function"""
    print("AMAZON MUSICAL INSTRUMENTS SENTIMENT ANALYSIS")
    print("=" * 60)
    print("Starting sentiment analysis pipeline...")

    # Initialize and run pipeline
    pipeline = SentimentAnalysisPipeline()

    try:
        pipeline.run_complete_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
