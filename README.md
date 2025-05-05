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

## Outputs
Feature Type: TF-IDF
```
Model: Random Forest
-------------------------
Accuracy  : 0.6583
Precision : 0.7052
Recall    : 0.6583
F1-Score  : 0.6509

Model: SVM
-------------------------
Accuracy  : 0.7708
Precision : 0.7939
Recall    : 0.7708
F1-Score  : 0.7654

Model: Naive Bayes
-------------------------
Accuracy  : 0.4958
Precision : 0.6306
Recall    : 0.4958
F1-Score  : 0.4590

Model: MLP
-------------------------
Accuracy  : 0.7583
Precision : 0.8178
Recall    : 0.7583
F1-Score  : 0.7478

Model: Decision Tree
-------------------------
Accuracy  : 0.3250
Precision : 0.3512
Recall    : 0.3250
F1-Score  : 0.3212

Model: XGBoost
-------------------------
Accuracy  : 0.6583
Precision : 0.7012
Recall    : 0.6583
F1-Score  : 0.6614
```
Feature Type: word_ngram
```
Model: Random Forest
-------------------------
Accuracy  : 0.4167
Precision : 0.5463
Recall    : 0.4167
F1-Score  : 0.4149

Model: SVM
-------------------------
Accuracy  : 0.6500
Precision : 0.6500
Recall    : 0.6500
F1-Score  : 0.6124

Model: Naive Bayes
-------------------------
Accuracy  : 0.6000
Precision : 0.6693
Recall    : 0.6000
F1-Score  : 0.5754

Model: MLP
-------------------------
Accuracy  : 0.2833
Precision : 0.5695
Recall    : 0.2833
F1-Score  : 0.3102

Model: Decision Tree
-------------------------
Accuracy  : 0.1750
Precision : 0.1694
Recall    : 0.1750
F1-Score  : 0.1588

Model: XGBoost
-------------------------
Accuracy  : 0.2583
Precision : 0.3804
Recall    : 0.2583
F1-Score  : 0.2585

```

Feature Type: char_ngram
```
Model: Random Forest
-------------------------
Accuracy  : 0.8792
Precision : 0.8960
Recall    : 0.8792
F1-Score  : 0.8763

Model: SVM
-------------------------
Accuracy  : 0.9292
Precision : 0.9373
Recall    : 0.9292
F1-Score  : 0.9290

Model: Naive Bayes
-------------------------
Accuracy  : 0.5375
Precision : 0.6713
Recall    : 0.5375
F1-Score  : 0.5249

Model: MLP
-------------------------
Accuracy  : 0.9417
Precision : 0.9495
Recall    : 0.9417
F1-Score  : 0.9416

Model: Decision Tree
-------------------------
Accuracy  : 0.6208
Precision : 0.6422
Recall    : 0.6208
F1-Score  : 0.6185

Model: XGBoost
-------------------------
Accuracy  : 0.8500
Precision : 0.8583
Recall    : 0.8500
F1-Score  : 0.8468

```
Feature Type: bert
```
Model: Random Forest
-------------------------
Accuracy  : 0.4375
Precision : 0.4035
Recall    : 0.4375
F1-Score  : 0.4122

Model: SVM
-------------------------
Accuracy  : 0.4125
Precision : 0.5079
Recall    : 0.4125
F1-Score  : 0.3960

Model: Naive Bayes
-------------------------
Accuracy  : 0.4750
Precision : 0.5301
Recall    : 0.4750
F1-Score  : 0.4718

Model: MLP
-------------------------
Accuracy  : 0.5500
Precision : 0.5690
Recall    : 0.5500
F1-Score  : 0.5442

Model: Decision Tree
-------------------------
Accuracy  : 0.2500
Precision : 0.2689
Recall    : 0.2500
F1-Score  : 0.2475

Model: XGBoost
-------------------------
Accuracy  : 0.4708
Precision : 0.4847
Recall    : 0.4708
F1-Score  : 0.4653

```
AVERAGE RESULTS ACROSS ALL FEATURE TYPES
```
Model: Random Forest
-------------------------
Accuracy  : 0.5979
Precision : 0.6378
Recall    : 0.5979
F1-Score  : 0.5886

Model: SVM
-------------------------
Accuracy  : 0.6906
Precision : 0.7223
Recall    : 0.6906
F1-Score  : 0.6757

Model: Naive Bayes
-------------------------
Accuracy  : 0.5271
Precision : 0.6253
Recall    : 0.5271
F1-Score  : 0.5078

Model: MLP
-------------------------
Accuracy  : 0.6333
Precision : 0.7265
Recall    : 0.6333
F1-Score  : 0.6359

Model: Decision Tree
-------------------------
Accuracy  : 0.3427
Precision : 0.3579
Recall    : 0.3427
F1-Score  : 0.3365

Model: XGBoost
-------------------------
Accuracy  : 0.5594
Precision : 0.6061
Recall    : 0.5594
F1-Score  : 0.5580


```
