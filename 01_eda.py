import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("Raisin_Dataset.xlsx")

print(df.shape)                    # kaç satır, kaç sütun
print(df.head())                   # ilk 5 satır
print(df.isnull().sum())           # eksik değer sayısı
print(df["Class"].value_counts())  # sınıf dağılımı


# İki sınıfı iki farklı renkle çiziyoruz
for sinif, renk in [("Kecimen", "tab:blue"), ("Besni", "tab:orange")]:
    alt = df[df["Class"] == sinif]
    plt.scatter(alt["Area"], alt["Eccentricity"],
                s=10, alpha=0.6, c=renk, label=sinif)

plt.xlabel("Area (Alan)")
plt.ylabel("Eccentricity (Dışmerkezlik)")
plt.title("Raisin veri seti: iki özelliğe göre sınıflar")
plt.legend()
plt.savefig("grafik_scatter.png", dpi=150)
plt.show()