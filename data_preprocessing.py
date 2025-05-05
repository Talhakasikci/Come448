import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data_from_folders(dataset_path):
    texts = []
    labels = []

    # Her yazar klasörünü dolaş
    for author in os.listdir(dataset_path):
        author_path = os.path.join(dataset_path, author)
        if os.path.isdir(author_path):  # Sadece klasörleri kontrol et
            for file in os.listdir(author_path):
                file_path = os.path.join(author_path, file)
                if os.path.isfile(file_path):  # Sadece dosyaları kontrol et
                    with open(file_path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                        labels.append(author)  # Yazar ismini etiket olarak kullan

    return pd.DataFrame({'text': texts, 'author': labels})

def preprocess_data(dataset_path):
    # Verileri yükle
    df = load_data_from_folders(dataset_path)

    # Eğitim ve test setlerini her yazar için ayrı ayrı ayır
    train_texts = []
    test_texts = []
    train_labels = []
    test_labels = []

    for author in df['author'].unique():
        author_texts = df[df['author'] == author]['text']
        author_labels = df[df['author'] == author]['author']

        # Yazarın yazılarını %80 eğitim, %20 test olarak ayır
        X_train, X_test, y_train, y_test = train_test_split(
            author_texts, author_labels, test_size=0.2, random_state=42
        )

        # Eğitim ve test verilerini birleştir
        train_texts.extend(X_train)
        test_texts.extend(X_test)
        train_labels.extend(y_train)
        test_labels.extend(y_test)

    # Etiketleri sayısal değerlere dönüştürmek için LabelEncoder kullan
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    return train_texts, test_texts, train_labels, test_labels