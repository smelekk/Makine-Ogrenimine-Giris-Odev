import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

class DogrusalRegresyonSiniflandirici:
    def __init__(self, ogrenme_orani=0.01, iterasyon_sayisi=1000, esik_degeri=0.5):
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon_sayisi = iterasyon_sayisi
        self.esik_degeri = esik_degeri
        self.agirliklar = None
        self.bias = None
        self.kayip_gecmisi = []
    def fit(self, X, y):
        ornek_sayisi, ozellik_sayisi  = X.shape
        self.agirliklar = np.zeros(ozellik_sayisi)
        self.bias = 0
        self.kayip_gecmisi = []  
        for i in range(self.iterasyon_sayisi):
            y_tahmin = np.dot(X, self.agirliklar) + self.bias # hiptoez: y = X.w + b
            mse = (1 / ornek_sayisi) * np.sum((y - y_tahmin) ** 2) # MSE = (1/m) * toplam((y_i - y_tahmin_i)^2)
            self.kayip_gecmisi.append(mse)
            #gradyanlar
            dw = (2/ornek_sayisi) * np.dot(X.T, (y_tahmin - y))
            db = (2/ornek_sayisi) * np.sum(y_tahmin - y)
            #ağırlık güncellemeleri
            self.agirliklar -= self.ogrenme_orani * dw 
            self.bias -= self.ogrenme_orani * db
    def predict_continuous(self, X):
        return np.dot(X, self.agirliklar) + self.bias 
    def predict(self, X):
        y_tahmin_surekli = self.predict_continuous(X)
        return (y_tahmin_surekli >= self.esik_degeri).astype(int)

# 1. VERİ YÜKLEME VE MODEL EĞİTİMİ
data = np.load("veri_hazir.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

model = DogrusalRegresyonSiniflandirici(ogrenme_orani=0.1, iterasyon_sayisi=1000)
model.fit(X_train, y_train)

# Tahminleri Alma
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
puan = model.predict_continuous(X_test)

print("=== DOĞRUSAL REGRESYON TEST PERFORMANSI ===")
print(f"Accuracy (Doğruluk):  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision (Kesinlik): {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall (Duyarlılık):  {recall_score(y_test, y_pred_test):.4f}")
print(f"F1 Score:             {f1_score(y_test, y_pred_test):.4f}")
print("-" * 40)
print(f"Eğitim Doğruluğu:     {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test Min / Max Puan:  {puan.min().round(2)} / {puan.max().round(2)}")

# KAYIP EĞRİSİ (LOSS CURVE)
plt.figure(figsize=(8, 4))
plt.plot(model.kayip_gecmisi, color='blue', linewidth=2)
plt.title("Doğrusal Regresyon - Öğrenme Eğrisi (MSE Loss)")
plt.xlabel("İterasyon (Epoch)")
plt.ylabel("Ortalama Kare Hata (MSE)")
plt.grid(True)
plt.savefig("dogrusal_kayip_egrisi.png", dpi=150)
plt.show()

# CONFUSION MATRIX (KARMAŞIKLIK MATRİSİ)
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Kecimen (0)', 'Besni (1)'],
            yticklabels=['Kecimen (0)', 'Besni (1)'])
plt.title("Doğrusal Regresyon - Karmaşıklık Matrisi (Confusion Matrix)")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.tight_layout()
plt.savefig("dogrusal_confusion_matrix.png", dpi=150)
plt.show()


