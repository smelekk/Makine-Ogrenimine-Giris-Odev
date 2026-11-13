import numpy as np
import pandas as pd

df = pd.read_excel("Raisin_Dataset.xlsx")
df["Class"] = df["Class"].map({"Kecimen": 0, "Besni": 1})

X = df.drop(columns="Class").to_numpy(dtype=float)
y = df["Class"].to_numpy(dtype=float)

# Eğitim / test -> %80 / %20
rng = np.random.default_rng(42)      
idx = rng.permutation(len(X))         
n_test = int(len(X) * 0.2)            
test_idx = idx[:n_test]
train_idx = idx[n_test:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Standartlaştırma: ortalama ve std sadece eğitimden hesaplanmıştır
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma         

# Kontrol çıktıları
print("Eğitim:", X_train.shape, " Test:", X_test.shape)
print("Eğitimde sınıf sayıları (0, 1):", np.bincount(y_train.astype(int)))
print("Testte sınıf sayıları (0, 1):  ", np.bincount(y_test.astype(int)))
print("Eğitim ortalaması:", X_train.mean(axis=0).round(2))
print("Eğitim std:       ", X_train.std(axis=0).round(2))
print("Test ortalaması:  ", X_test.mean(axis=0).round(2))

# np.savez("veri_hazir.npz", X_train=X_train, X_test=X_test,
#          y_train=y_train, y_test=y_test)