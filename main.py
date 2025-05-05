from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model_training import train_models
from evaluation import evaluate_models
import pandas as pd
import numpy as np

# Veri seti yolu
dataset_path = "d:\\Come448\\dataset_authorship"

# 1. Veri yükleme ve ön işleme
X_train, X_test, y_train, y_test = preprocess_data(dataset_path)

# 2. Özellik çıkarma
features = extract_features(X_train, X_test)

# 3. Her özellik türü için modelleri eğit ve değerlendir
results_all = {}

feature_types = ["TF-IDF", "word_ngram", "char_ngram", "bert"]  # Feature type listesi

for feature_type in feature_types:
    print(f"\nProcessing {feature_type} features...")
    trained_models = train_models(features[feature_type][0], y_train)
    results = evaluate_models(trained_models, features[feature_type][1], y_test)
    results_all[feature_type] = results

# 4. Sonuçları yazdır ve kaydet
print("\nClassification Results:")
print("=" * 80)

# Her feature type için sonuçları yazdır
for feature_type, results in results_all.items():
    print(f"\nFeature Type: {feature_type}")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 25)
        for metric_name, value in metrics.items():
            print(f"{metric_name:10}: {value:.4f}")

# Her model için ortalama hesapla
print("\nAVERAGE RESULTS ACROSS ALL FEATURE TYPES")
print("=" * 80)

model_averages = {}
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Her model için
for model_name in results_all["TF-IDF"].keys():  # "word" yerine "TF-IDF" kullan
    model_metrics = {metric: [] for metric in metric_names}
    
    # Her feature type'dan metrikleri topla
    for feature_type in feature_types:  # feature_types listesini kullan
        for metric in metric_names:
            model_metrics[metric].append(results_all[feature_type][model_name][metric])
    
    # Ortalamalar
    print(f"\nModel: {model_name}")
    print("-" * 25)
    for metric in metric_names:
        avg_value = np.mean(model_metrics[metric])
        print(f"{metric:10}: {avg_value:.4f}")

# Results to DataFrame for easy analysis
results_df = pd.DataFrame()
for feature_type, results in results_all.items():
    for model_name, metrics in results.items():
        row = pd.DataFrame({
            'Feature_Type': [feature_type],
            'Model': [model_name],
            **{k: [v] for k, v in metrics.items()}
        })
        results_df = pd.concat([results_df, row], ignore_index=True)

# Save results to CSV
results_df.to_csv('classification_results.csv', index=False)
print("\nResults have been saved to 'classification_results.csv'")