import pandas as pd
import numpy as np
from collections import Counter

# Verisetini yükleme
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Veri ön işleme
# Gereksiz sütunların çıkarılması
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

# 'diagnosis' sütununu kodlama (kötü huylu = 1, iyi huylu = 0)
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Özelliklerin (X) ve hedef değişkenin (y) ayrılması
X = data.drop(['diagnosis'], axis=1).values
y = data['diagnosis'].values

# Veri setinin eğitim ve test setlerine bölünmesi
def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    size = int(len(X) * (1 - test_size))
    return X[:size], X[size:], y[:size], y[size:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizasyon işleminin yapılması
def normalize(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    return (X - min_vals) / (max_vals - min_vals)

# Eğitim ve test verilerinin normalizasyonu
X_train = normalize(X_train)
X_test = normalize(X_test)

# K-En Yakın Komşu Algoritması
class KNearestNeighbors:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # x ve eğitim setindeki tüm örnekler arasındaki Öklid mesafesini hesaplama
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Mesafeye göre sıralama ve ilk k komşunun indekslerini alma
        k_indices = np.argsort(distances)[:self.k]
        # k en yakın komşunun etiketlerini alma
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # En yaygın sınıf etiketini döndürme
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Doğruluk puanını hesaplama
def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# Sınıflandırma raporu oluşturma
def classification_report(y_true, y_pred):
    precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

# Modelin eğitilmesi ve test edilmesi
knn = KNearestNeighbors(k=2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Modelin değerlendirilmesi
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f'Doğruluk Oranı: {accuracy}')
print('Sınıflandırma Raporu:')
print(report)
