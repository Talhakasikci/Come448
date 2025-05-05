from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def get_bert_features(texts, model, tokenizer):
    features = []
    with torch.no_grad():
        for text in texts:
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get BERT embeddings
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as text representation
            features.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
    
    return np.vstack(features)

def extract_features(X_train, X_test):
    # Initialize BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Get BERT features
    X_train_bert = get_bert_features(X_train, model, tokenizer)
    X_test_bert = get_bert_features(X_test, model, tokenizer)

    # TF-IDF (kelime tabanlı) ve n-gram özellik çıkarma
    tfidf_vectorizer_word = TfidfVectorizer(ngram_range=(1, 1))  # Unigram
    tfidf_vectorizer_word_ngram = TfidfVectorizer(ngram_range=(2, 3))  # 2-gram ve 3-gram
    tfidf_vectorizer_char_ngram = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))  # Karakter bazlı

    # Eğitim seti için TF-IDF matrislerini oluştur
    X_train_tfidf_word = tfidf_vectorizer_word.fit_transform(X_train)
    X_train_tfidf_word_ngram = tfidf_vectorizer_word_ngram.fit_transform(X_train)
    X_train_tfidf_char_ngram = tfidf_vectorizer_char_ngram.fit_transform(X_train)

    # Test seti için TF-IDF matrislerini oluştur
    X_test_tfidf_word = tfidf_vectorizer_word.transform(X_test)
    X_test_tfidf_word_ngram = tfidf_vectorizer_word_ngram.transform(X_test)
    X_test_tfidf_char_ngram = tfidf_vectorizer_char_ngram.transform(X_test)

    return {
        "TF-IDF": (X_train_tfidf_word, X_test_tfidf_word),
        "word_ngram": (X_train_tfidf_word_ngram, X_test_tfidf_word_ngram),
        "char_ngram": (X_train_tfidf_char_ngram, X_test_tfidf_char_ngram),
        "bert": (X_train_bert, X_test_bert)
    }