from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import scipy.sparse as sp
import numpy as np

def train_models(X_train, y_train):
    # Veri tipini kontrol et
    is_sparse = sp.issparse(X_train)
    has_negative = (X_train < 0).any() if not is_sparse else False

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        # BERT için GaussianNB, diğerleri için MultinomialNB kullan
        "Naive Bayes": GaussianNB() if has_negative else MultinomialNB(),
        "MLP": MLPClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42)
    }

    # Eğer sparse matrix ise ve GaussianNB kullanılıyorsa, yoğun (dense) matrise çevir
    if is_sparse and has_negative:
        X_train = X_train.toarray()

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models