import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




def saveHeatMap(X, column_names):
    """ Generacja i zapisanie heatmap

    Args:
        X (matrix): Macież cech
        column_names (array): tablica nazw wszystkich kolumn
    """
    corelation = np.corrcoef(X, rowvar=False)

    plt.figure(figsize=(10, 8))

    sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, xticklabels=column_names[:-1], yticklabels=column_names[:-1])
    plt.title('Macierz korelacji cech środowiskowych')
    plt.tight_layout()
    plt.savefig('heatmapa.png')

def checkData(X):
    """ Gprawdzenie danych

    Args:
        X (matrix): Macież cech
    """
    print("\nBRAKI DANYCH")

    total_missing = np.isnan(X).sum()
    print(f"Łączna liczba pustych miejsc w cechach (X): {total_missing}")
    if total_missing == 0:
        print("Zbiór cech jest kompletny, nie ma żadnych pustych komórek")
    else:
        print("Znaleziono braki w danych")

def checkClassBinalse(y):
    """ Sprawdzenie zbalansowania klas

    Args:
        y (matrix): Macież klas
    """

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