"""
Basic tests for sentiment analysis pipeline
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.text_preprocessing import TextPreprocessor

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_data = pd.DataFrame({
            'reviewerID': ['A123', 'B456'],
            'asin': ['123456789', '987654321'],
            'reviewerName': ['John', 'Jane'],
            'helpful': ['[1, 2]', '[0, 1]'],
            'reviewText': [
                'This product is amazing! Great quality.',
                'Not very good. Disappointed with the purchase.'
            ],
            'overall': [5.0, 2.0],
            'summary': ['Great product', 'Poor quality'],
            'unixReviewTime': [1234567890, 1234567891],
            'reviewTime': ['01 15, 2010', '01 16, 2010']
        })

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        self.assertIsInstance(engineer, FeatureEngineer)
        self.assertEqual(len(engineer.feature_names), 0)
        self.assertFalse(engineer.is_fitted)

    def test_feature_engineering_transform(self):
        """Test feature engineering transform"""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.sample_data)

        # Check that new features are added
        self.assertIn('sentiment_binary', result.columns)
        self.assertIn('sentiment_3class', result.columns)
        self.assertIn('helpful_votes', result.columns)
        self.assertIn('review_length', result.columns)

        # Check target variables
        self.assertEqual(result.iloc[0]['sentiment_binary'], 1)  # 5-star -> positive
        self.assertEqual(result.iloc[1]['sentiment_binary'], 0)  # 2-star -> negative

class TestTextPreprocessing(unittest.TestCase):
    """Test text preprocessing functionality"""

    def setUp(self):
        """Set up test data"""
        self.sample_texts = [
            "This is a GREAT product! I love it.",
            "Terrible quality... Don't buy this!",
            "It's okay, nothing special."
        ]

    def test_preprocessor_initialization(self):
        """Test TextPreprocessor initialization"""
        preprocessor = TextPreprocessor()
        self.assertIsInstance(preprocessor, TextPreprocessor)
        self.assertFalse(preprocessor.is_fitted)

    def test_text_preprocessing(self):
        """Test basic text preprocessing"""
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_digits=True
        )

        result = preprocessor.fit_transform(self.sample_texts)

        # Check that processing occurred
        self.assertTrue(preprocessor.is_fitted)
        self.assertEqual(len(result), len(self.sample_texts))

        # Check lowercase conversion
        for text in result:
            self.assertEqual(text, text.lower())

    def test_empty_text_handling(self):
        """Test handling of empty texts"""
        preprocessor = TextPreprocessor()
        empty_texts = ['', None, '   ']

        result = preprocessor.fit_transform(empty_texts)

        # All should become empty strings
        for text in result:
            self.assertEqual(text.strip(), '')

if __name__ == '__main__':
    unittest.main()
