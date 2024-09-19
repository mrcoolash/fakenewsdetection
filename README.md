# Fake News Detection

## Overview
This project aims to classify news articles as real or fake using various machine learning algorithms. It employs text processing techniques and multiple classifiers to detect fake news with high accuracy.

## Features
- Text preprocessing and vectorization using TF-IDF.
- Multiple machine learning models implemented:
  - Logistic Regression (LR)
  - Gradient Boosting Classifier (GBC)
  - Random Forest Classifier (RFC)
- Model evaluation with precision, recall, F1-score, and accuracy metrics.

## Dataset
The dataset used for training and testing consists of labeled news articles (real and fake). The dataset was sourced from Kaggle's Fake News dataset.

## Installation
To run the project, ensure you have Python 3 and the required dependencies installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/mrcoolash/fakenewsdetection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd fakenewsdetection
   ```

3. Install dependencies using requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Data Preprocessing: The dataset is preprocessed to remove noise and vectorized using the TF-IDF vectorizer.
2. Model Training: The following models are trained:
   - Logistic Regression
   - Gradient Boosting Classifier
   - Random Forest Classifier
3. Prediction: You can run predictions on new data using the trained models.
4. Evaluation: Model performance is evaluated based on accuracy, precision, recall, and F1-score.

To train and test the model:
```bash
# Run the notebook
jupyter notebook fakenewsdetection.ipynb
```

## Results
- Logistic Regression: Accuracy ~ 98%
- Gradient Boosting Classifier: Accuracy ~ 97%
- Random Forest Classifier: Accuracy ~ 99%

## Project Structure
```
fakenews-detection/
│
├── data/                   # Dataset files
├── fakenewsdetection.ipynb  # Jupyter notebook containing code
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
└── model/                   # Saved machine learning models
```

## Dependencies
- Python 3.x
- Pandas
- Scikit-learn
- Numpy
- Jupyter

## Conclusion
This project demonstrates how machine learning techniques can be applied to detect fake news articles with high accuracy. Feel free to explore the code and experiment with different classifiers or datasets to improve performance.
