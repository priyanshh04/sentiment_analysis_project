# Amazon Musical Instruments Reviews - Sentiment Analysis

A comprehensive machine learning pipeline for sentiment analysis of Amazon Musical Instruments Reviews dataset.

## 🎯 Project Overview

This project implements an end-to-end sentiment analysis solution that:
- Processes 10,261 Amazon musical instrument reviews from 2004-2014
- Handles severe class imbalance (87.9% positive reviews)
- Engineers 12 additional features beyond raw text
- Trains and compares 5 different ML algorithms
- Provides comprehensive evaluation and visualization tools

## 📊 Dataset Information

- **Total Reviews:** 10,261
- **Time Period:** 2004-2014
- **Unique Reviewers:** 1,429
- **Unique Products:** 900
- **Average Review Length:** 90 words
- **Class Distribution:** 12.1% negative, 87.9% positive

## 🏗️ Project Structure

```
sentiment_analysis_project/
├── configs/
│   └── config.py                 # Project configuration
├── data/
│   ├── raw/                     # Raw dataset
│   ├── processed/               # Processed dataset
│   └── models/                  # Saved models
├── src/
│   ├── utils/
│   │   └── data_loader.py       # Data loading utilities
│   ├── preprocessing/
│   │   ├── feature_engineering.py  # Feature engineering
│   │   └── text_preprocessing.py   # Text preprocessing
│   ├── models/
│   │   └── sentiment_models.py     # ML model implementations
│   └── evaluation/
│       └── model_evaluation.py     # Evaluation metrics and plots
├── notebooks/
│   └── sentiment_analysis_exploration.ipynb  # Interactive analysis
├── requirements/
│   └── requirements.txt         # Python dependencies
├── tests/                       # Unit tests
├── main.py                     # Main pipeline script
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd sentiment_analysis_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements/requirements.txt
```

### 2. Data Preparation

Place your `Instruments_Reviews.csv` file in the `data/raw/` directory.

### 3. Run Complete Pipeline

```bash
# Run the complete sentiment analysis pipeline
python main.py
```

This will:
1. Load and validate the dataset
2. Apply feature engineering
3. Preprocess text data
4. Train all ML models
5. Evaluate and compare models
6. Save results and visualizations

### 4. Interactive Exploration

```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks/sentiment_analysis_exploration.ipynb
```

## 🔧 Configuration

Modify `configs/config.py` to customize:
- Text preprocessing parameters
- Model training settings
- Feature engineering options
- Evaluation metrics

## 🤖 Machine Learning Pipeline

### 1. Feature Engineering
- **Target Variables:** Binary and 3-class sentiment labels
- **Text Features:** Length, word count, punctuation analysis
- **Helpfulness Metrics:** Extracted from user votes
- **Temporal Features:** Year, month, seasonal patterns
- **Sentiment Indicators:** Emotion markers, capitalization

### 2. Text Preprocessing
- Lowercase conversion
- URL and HTML removal
- Punctuation and digit removal
- Stopword removal
- Lemmatization
- Custom domain-specific cleaning

### 3. Model Training
- **Logistic Regression:** Fast, interpretable baseline
- **Support Vector Machine:** High-dimensional text classification
- **Random Forest:** Ensemble method for robustness
- **Naive Bayes:** Traditional text classification
- **XGBoost:** Advanced gradient boosting (optional)

### 4. Class Imbalance Handling
- SMOTE (Synthetic Minority Oversampling)
- Class weight balancing
- Stratified sampling
- Focus on minority class metrics

### 5. Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Cross-validation
- Per-class performance analysis

## 📈 Results

The pipeline typically achieves:
- **Accuracy:** 85-90%
- **F1-Score:** 0.75-0.85
- **Best Model:** Usually Logistic Regression or SVM

Detailed results are saved in:
- `reports/training_results.json`
- `reports/evaluation_report.txt`
- `plots/` directory with visualizations

## 📁 Output Files

After running the pipeline, you'll find:

### Data Files
- `data/processed/processed_instruments_reviews.csv` - Engineered dataset
- `data/models/` - Trained model files

### Reports
- `reports/training_results.json` - Detailed training metrics
- `reports/evaluation_report.txt` - Comprehensive evaluation report
- `sentiment_analysis.log` - Execution logs

### Visualizations
- `plots/confusion_matrix_*.png` - Confusion matrices
- `plots/roc_curves_comparison.png` - ROC curve comparison
- `plots/metrics_comparison.png` - Model performance comparison

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py -v
```

## 📚 Key Features

### Advanced Text Preprocessing
- Domain-specific stopword handling
- Contraction expansion
- Lemmatization with POS tagging
- Configurable cleaning pipeline

### Comprehensive Feature Engineering
- 12 engineered features beyond raw text
- Temporal analysis capabilities
- Social validation metrics (helpfulness)
- Emotion and sentiment indicators

### Robust Model Training
- Automated hyperparameter exploration
- Cross-validation with stratified sampling
- SMOTE integration for imbalanced data
- Model persistence and loading

### Rich Evaluation Suite
- Multiple evaluation metrics
- Visualization generation
- Comparative analysis tools
- Business-focused reporting

## 🔍 Usage Examples

### Basic Usage

```python
from main import SentimentAnalysisPipeline

# Initialize pipeline
pipeline = SentimentAnalysisPipeline()

# Run complete analysis
pipeline.run_complete_pipeline()
```

### Custom Model Training

```python
from src.models.sentiment_models import SentimentModelTrainer

# Initialize trainer
trainer = SentimentModelTrainer(use_smote=True)

# Prepare data
trainer.prepare_data(df, 'reviewText', 'sentiment_binary')

# Train specific model
results = trainer.train_model('logistic_regression')
```

### Text Preprocessing Only

```python
from src.preprocessing.text_preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(
    remove_stopwords=True,
    apply_lemmatization=True
)

# Process texts
clean_texts = preprocessor.fit_transform(raw_texts)
```

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Memory Issues**
   - Reduce dataset size for testing
   - Adjust TF-IDF max_features parameter
   - Use smaller CV folds

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify file structure

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Amazon for providing the musical instruments review dataset
- scikit-learn team for excellent ML library
- NLTK and spaCy teams for NLP capabilities
- Contributors and the open-source community

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation in `notebooks/`

---

**Happy Sentiment Analysis! 🎵📊**
