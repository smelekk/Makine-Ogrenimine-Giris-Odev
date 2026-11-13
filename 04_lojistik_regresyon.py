import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

class LojistikRegresyonSiniflandirici:
    def __init__(self, ogrenme_orani=0.01, iterasyon_sayisi=1000, esik_degeri=0.5):
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon_sayisi = iterasyon_sayisi
        self.esik_degeri = esik_degeri
        self.agirliklar = None
        self.bias = None
        self.kayip_gecmisi = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        ornek_sayisi, ozellik_sayisi = X.shape
        self.agirliklar = np.zeros(ozellik_sayisi)
        self.bias = 0
        self.kayip_gecmisi = []
        for i in range(self.iterasyon_sayisi):
            z = np.dot(X, self.agirliklar) + self.bias
            y_tahmin = self._sigmoid(z)   #çıktı artık [0,1] arasında
            epsilon = 1e-15
            y_tahmin_kirpilmis = np.clip(y_tahmin, epsilon, 1 - epsilon)
            log_loss = -(1 / ornek_sayisi) * np.sum( y * np.log(y_tahmin_kirpilmis) + (1 - y) * np.log(1 - y_tahmin_kirpilmis))
            self.kayip_gecmisi.append(log_loss)

            dw = (1 / ornek_sayisi) * np.dot(X.T, (y_tahmin - y))
            db = (1 / ornek_sayisi) * np.sum(y_tahmin - y)

            self.agirliklar -= self.ogrenme_orani * dw
            self.bias -= self.ogrenme_orani * db

    def predict_proba(self, X):
        z = np.dot(X, self.agirliklar) + self.bias
        return self._sigmoid(z)
    def predict(self, X):
        y_olasilik = self.predict_proba(X)
        return (y_olasilik >= self.esik_degeri).astype(int)


# VERİ YÜKLEME VE MODEL EĞİTİMİ
data = np.load("veri_hazir.npz")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

model = LojistikRegresyonSiniflandirici(ogrenme_orani=0.1, iterasyon_sayisi=1000)
model.fit(X_train, y_train)

# Tahminler
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
olasiliklar = model.predict_proba(X_test)

# PERFORMANS METRİKLERİ
print("=== LOJİSTİK REGRESYON TEST PERFORMANSI ===")
print(f"Accuracy (Doğruluk):  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision (Kesinlik): {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall (Duyarlılık):  {recall_score(y_test, y_pred_test):.4f}")
print(f"F1 Score:             {f1_score(y_test, y_pred_test):.4f}")
print("-" * 40)
print(f"Eğitim Doğruluğu:     {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Test Min / Max Olasılık: {olasiliklar.min().round(4)} / {olasiliklar.max().round(4)}")

# GÖRSELLEŞTİRME
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Kayıp Eğrisi (Log-Loss)
axes[0].plot(model.kayip_gecmisi, color='green', linewidth=2)
axes[0].set_title("Lojistik Regresyon - Öğrenme Eğrisi (Log-Loss)")
axes[0].set_xlabel("İterasyon (Epoch)")
axes[0].set_ylabel("Binary Cross-Entropy Loss")
axes[0].grid(True)

# 2. Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Kecimen (0)', 'Besni (1)'],
            yticklabels=['Kecimen (0)', 'Besni (1)'])
axes[1].set_title("Lojistik Regresyon - Karmaşıklık Matrisi")
axes[1].set_xlabel("Tahmin Edilen Sınıf")
axes[1].set_ylabel("Gerçek Sınıf")

plt.tight_layout()
plt.savefig("lojistik_performans_grafikleri.png", dpi=150)
plt.show()