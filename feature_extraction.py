from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(X_train, X_test):
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
        "word": (X_train_tfidf_word, X_test_tfidf_word),
        "word_ngram": (X_train_tfidf_word_ngram, X_test_tfidf_word_ngram),
        "char_ngram": (X_train_tfidf_char_ngram, X_test_tfidf_char_ngram)
    }