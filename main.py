from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from model_training import train_models
from evaluation import evaluate_models

# Veri seti yolu
dataset_path = "d:\\Come448\\dataset_authorship"

# 1. Veri yükleme ve ön işleme
X_train, X_test, y_train, y_test = preprocess_data(dataset_path)

# 2. Özellik çıkarma
features = extract_features(X_train, X_test)

# 3. Model eğitimi (örnek olarak kelime tabanlı TF-IDF kullanıyoruz)
trained_models = train_models(features["word"][0], y_train)

# 4. Model değerlendirme
results = evaluate_models(trained_models, features["word"][1], y_test)

# 5. Sonuçları yazdır
print("\nClassification Results:")
print("-" * 50)
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print("-" * 25)
    for metric_name, value in metrics.items():
        print(f"{metric_name:10}: {value:.4f}")