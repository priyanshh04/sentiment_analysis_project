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
├── requirements.txt             # Python dependencies 
├── tests/                       # Unit tests
│   └── test_pipeline.py
├── main.py                     # Main pipeline script
└── README.md                   # This file
```

## 💻 System Requirements & Environment

### Recommended Development Environment
For the smoothest experience, run this project using:
- **VS Code** (Visual Studio Code) - Recommended
- **PyCharm Professional/Community**
- **Jupyter Lab**
- **Google Colab** (for cloud-based execution)
- **Any Python-compatible IDE** with terminal support

### System Requirements
- **Python:** 3.8 or higher
- **RAM:** Minimum 4GB, Recommended 8GB+
- **Storage:** 2GB free space for models and outputs
- **OS:** Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### Why Use VS Code?
✅ **Integrated Terminal** - Run commands without switching windows  
✅ **Built-in Image Viewer** - View generated plots directly in the editor  
✅ **Python Extension** - Excellent debugging and IntelliSense  
✅ **Jupyter Integration** - Run notebooks seamlessly  
✅ **Git Integration** - Version control made easy  
✅ **Extensions Ecosystem** - CSV viewers, data visualization tools  

## 🚀 Quick Start

### 1. Environment Setup

**Option A: VS Code Setup (Recommended)**
```bash
# Open VS Code
# File > Open Folder > Select sentiment_analysis_project

# Open integrated terminal (Ctrl+` or Terminal > New Terminal)
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Command Line Setup**
```bash
# Navigate to project directory
cd sentiment_analysis_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option C: Google Colab (Cloud-based)**
```python
# Upload your project files to Colab
# Run this in a Colab cell:
!pip install -r requirements.txt
```

### 2. Data Preparation

Place your `Instruments_Reviews.csv` file in the `data/raw/` directory.

### 3. Run Complete Pipeline

**In VS Code:**
```bash
# Use integrated terminal (Ctrl+`)
python main.py
```

**In other IDEs/Command Line:**
```bash
python main.py
```

**In Google Colab:**
```python
# In a Colab cell
!python main.py
```

This will:
1. Load and validate the dataset
2. Apply feature engineering
3. Preprocess text data
4. Train all ML models
5. Evaluate and compare models
6. Save results and visualizations

### 4. Interactive Exploration

**VS Code with Jupyter Extension:**
- Install Python and Jupyter extensions in VS Code
- Open `notebooks/sentiment_analysis_exploration.ipynb`
- Click "Run Cell" to execute interactively

**Traditional Jupyter:**
```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks/sentiment_analysis_exploration.ipynb
```

**Google Colab:**
- Upload the notebook file to Google Colab
- Run cells interactively in the browser

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

### 3. Model Training & Performance

| Model | Training Time | Accuracy | F1-Score | Notes |
|-------|---------------|----------|----------|-------|
| **Logistic Regression** | 30-60 seconds | 85-88% | 0.75-0.80 | Fast, interpretable baseline |
| **Naive Bayes** | 15-30 seconds | 82-85% | 0.70-0.75 | Traditional text classification |
| **Random Forest** | 2-5 minutes | 84-87% | 0.73-0.78 | Ensemble method for robustness |
| **Support Vector Machine** | 10-15 minutes | 86-90% | 0.76-0.82 | High-dimensional text classification |
| **XGBoost** | 3-8 minutes | 87-90% | 0.77-0.83 | Advanced gradient boosting (optional) |

**Total Pipeline Runtime:** 15-25 minutes (including preprocessing and evaluation)

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
- **Overall Accuracy:** 85-90%
- **Overall F1-Score:** 0.75-0.85
- **Best Model:** Usually Logistic Regression or SVM
- **Training Time:** 15-25 minutes total

### Model Performance Comparison

| Metric | Logistic Regression | SVM | Random Forest | Naive Bayes | XGBoost |
|--------|-------------------|-----|---------------|-------------|---------|
| **Accuracy** | 87.6% | 88.4% | 86.1% | 83.2% | 88.9% |
| **F1-Score** | 0.78 | 0.81 | 0.75 | 0.72 | 0.82 |
| **Training Time** | 45 sec | 12 min | 4 min | 25 sec | 6 min |
| **Speed Rating** | ⚡⚡⚡ | ⚡ | ⚡⚡ | ⚡⚡⚡ | ⚡⚡ |

Detailed results are saved in:
- `reports/training_results.json`
- `reports/evaluation_report.txt`
- `plots/` directory with visualizations

## ⏱️ Performance & Timing

### Expected Runtime on Standard Hardware

| Component | Duration | Details |
|-----------|----------|---------|
| **Data Loading** | 5-15 seconds | Loading 10K+ reviews |
| **Feature Engineering** | 30-60 seconds | Creating 12 new features |
| **Text Preprocessing** | 2-5 minutes | NLTK tokenization, lemmatization |
| **Model Training** | 15-25 minutes | All 5 models with cross-validation |
| **Evaluation & Plots** | 1-3 minutes | Metrics calculation, visualization |
| **Results Saving** | 10-30 seconds | Model files, reports, plots |

**📝 Note:** SVM takes the longest (10-15 minutes) due to high-dimensional text features and probability calculations. For faster development iterations, consider training only Logistic Regression and Naive Bayes first.

## 📁 Output Files

After running the pipeline, you'll find:

### Data Files
- `data/processed/processed_instruments_reviews.csv` - Engineered dataset
- `data/models/` - Trained model files

### Reports
- `reports/training_results.json` - Detailed training metrics
- `reports/evaluation_report.txt` - Comprehensive evaluation report
- `sentiment_analysis.log` - Execution logs

### Visualizations (Viewable in VS Code!)
- `plots/confusion_matrix_*.png` - Confusion matrices
- `plots/roc_curves_comparison.png` - ROC curve comparison
- `plots/metrics_comparison.png` - Model performance comparison

**💡 Pro Tip:** In VS Code, you can right-click on any `.png` file in the plots folder and select "Open Preview" to view the generated visualizations directly in the editor!

## 🧪 Testing

**In VS Code Terminal:**
```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_preprocessing.py -v
```

**Command Line:**
```bash
python -m pytest tests/
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

4. **SVM Taking Too Long**
   - Use `kernel='linear'` for faster training
   - Reduce cross-validation folds during development
   - Consider using LinearSVC for production

5. **Plots Not Opening in VS Code**
   - Install Python extension for VS Code
   - Right-click PNG files and select "Open Preview"
   - Ensure matplotlib backend is set correctly

6. **Virtual Environment Issues**
   ```bash
   # If activation fails, try:
   python -m pip install --user virtualenv
   python -m virtualenv venv
   ```

## 💡 Optimization Tips

### For Faster Development
- **Use VS Code** with integrated terminal for seamless workflow
- Train only Logistic Regression and Naive Bayes initially
- Use smaller data samples (e.g., 2000 reviews) for testing
- Reduce cross-validation folds to 3 instead of 5
- Set `use_smote=False` for quicker iteration

### For Production
- Use all models for comprehensive comparison
- Enable full cross-validation for robust evaluation
- Implement model ensemble for improved performance
- Monitor training time and adjust parameters accordingly

### IDE-Specific Tips

**VS Code Users:**
- Use the integrated terminal (`Ctrl+``) for running scripts
- Install the Python and Jupyter extensions
- Use the built-in Git integration for version control
- Enable auto-save to prevent data loss during long training runs

**PyCharm Users:**
- Configure the Python interpreter to use your virtual environment
- Use the built-in package manager for dependency installation
- Leverage the powerful debugging capabilities

**Google Colab Users:**
- Mount Google Drive for persistent storage
- Use GPU runtime for faster model training (if available)
- Save important outputs to Drive before session expires

## 📝 Development Environment Setup

### VS Code Extensions (Recommended)
```
1. Python (by Microsoft)
2. Jupyter (by Microsoft)
3. CSV Viewer (by GrapeCity)
4. GitLens (by GitKraken)
5. Pylance (by Microsoft)
```

### PyCharm Setup
- Install PyCharm Community (free) or Professional
- Configure interpreter: Settings > Project > Python Interpreter
- Install packages through the GUI or terminal

### Command Line Tools
```bash
# Essential tools
pip install jupyter notebook
pip install pytest
pip install black  # Code formatter
pip install flake8  # Code linter
```

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

## 🎯 Final Notes

**For the best experience:**
1. **Use VS Code** with Python and Jupyter extensions installed
2. **Activate your virtual environment** before running any scripts
3. **Check the plots folder** after running - visualizations will be saved as PNG files
4. **Monitor the terminal output** for progress updates during training
5. **Be patient with SVM training** - it takes 10-15 minutes but produces excellent results!

---

**Happy Sentiment Analysis! 🎵📊**

*Developed with ❤️ for seamless execution in VS Code and other modern IDEs*
