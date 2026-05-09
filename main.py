import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sekcja 1: wczytanie danych
data = np.loadtxt("data/updated_pollution_dataset.csv", delimiter=",", dtype="object")
print(f"\nKształt danych (wiersz, kolumny): {data.shape}")

column_names = data[0]
print("\nNazwy kolumn:")
print(column_names)

data = data[1:]
print(f"\nKształt danych po ucięciu nagłówka: {data.shape}")

X = data[:, 0:9].astype(float)
y = data[:, -1]

print(f"\nKształt X (cechy): {X.shape}")
print(f"Kształt y (etykiety): {y.shape}")


# Sekcja 2: sprawdzenie zbalansowania klas
unique_classes, counts = np.unique(y, return_counts=True)

print("\nBALANS KLAS")
for cls, count in zip(unique_classes, counts):
    print(f"{cls}: {count}")

colors = ['green', 'red', 'gold', 'orange']

plt.figure(figsize=(8, 5))
plt.bar(unique_classes, counts, color=colors)
plt.title('Rozkład klas jakości powietrza (Balans klas)')
plt.xlabel('Jakość powietrza')
plt.ylabel('Liczba próbek')
plt.grid(axis='y')
plt.savefig('balans.png')

# Sekcja 3: sprawdzenie danych

print("\nBRAKI DANYCH")

total_missing = np.isnan(X).sum()
print(f"Łączna liczba pustych miejsc w cechach (X): {total_missing}")
if total_missing == 0:
    print("Zbiór cech jest kompletny, nie ma żadnych pustych komórek")
else:
    print("Znaleziono braki w danych")

# Sekcja 4: wizualizacja korelacji cech

corelation = np.corrcoef(X, rowvar=False)

plt.figure(figsize=(10, 8))

sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, xticklabels=column_names[:-1], yticklabels=column_names[:-1])
plt.title('Macierz korelacji cech środowiskowych')
plt.tight_layout()
plt.savefig('heatmapa.png')