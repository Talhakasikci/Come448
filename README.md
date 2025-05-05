# Author Classification Project

## Overview
This project implements a text classification system to identify authors based on their writing style using various machine learning techniques. The system uses TF-IDF feature extraction and multiple classification algorithms to determine authorship of text documents.

## Features
- Text preprocessing and feature extraction using TF-IDF
- Multiple classification algorithms:
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Multi-Layer Perceptron (MLP)
  - Decision Tree
  - XGBoost

## Project Structure
```
d:\Come448\
├── dataset_authorship\    # Dataset directory containing author folders
├── data_preprocessing.py  # Data loading and preprocessing
├── feature_extraction.py  # TF-IDF feature extraction
├── model_training.py     # Training multiple classifiers
├── evaluation.py         # Model evaluation metrics
└── main.py              # Main execution script
```

## Requirements
- Python 3.x
- scikit-learn
- pandas
- xgboost

## Usage
1. Ensure all required libraries are installed:
```bash
pip install pandas scikit-learn xgboost
```

2. Run the main script:
```bash
python main.py
```

## Performance Metrics
The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-Score

## Dataset
The dataset consists of text documents organized in folders by author, with an 80-20 train-test split for each author's documents.

![alt text](<Ekran görüntüsü 2025-05-05 141614.png>)