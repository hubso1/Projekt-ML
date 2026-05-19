import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def saveHeatMap(X, column_names):
    """ Generacja i zapisanie heatmap

    Args:
        X (matrix): Macierz cech
        column_names (array): tablica nazw wszystkich kolumn
    """
    corelation = np.corrcoef(X, rowvar=False)

    plt.figure(figsize=(10, 8))

    sns.heatmap(corelation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, xticklabels=column_names[:-1], yticklabels=column_names[:-1])
    plt.title('Macierz korelacji cech środowiskowych')
    plt.tight_layout()
    plt.savefig('heatmapa.png')

def checkData(X):
    """ Sprawdzenie danych

    Args:
        X (matrix): Macierz cech
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
        y (matrix): Macierz klas
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


def saveResamplingBarChart(baseline_results, resampling_results, classifiers_names):
    """ Generowanie grupowego wykresu słupkowego porównujący wyniki resamplingu.

    Args:
        baseline_results (array): Wyniki bez resamplingu.
        resampling_results (dict): Słownik z wynikami po resamplingu.
        classifiers_names (list): Nazwy testowanych klasyfikatorów.
    """
    all_results = []

    for index, clf_name in enumerate(classifiers_names):
        mean_bac = np.mean(baseline_results[index])
        all_results.append({'Klasyfikator': clf_name, 'Metoda': 'Baseline', 'BAC': mean_bac})

    for combo_name, scores in resampling_results.items():
        clf_name, method_name = combo_name.split(' + ')
        all_results.append({'Klasyfikator': clf_name, 'Metoda': method_name, 'BAC': np.mean(scores)})

    df = pd.DataFrame(all_results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Klasyfikator', y='BAC', hue='Metoda', palette='viridis')

    plt.title('Porównanie skuteczności klasyfikatorów z różnymi metodami resamplingu')
    plt.ylabel('Balanced Accuracy (BAC)')
    plt.ylim(0, 1.1)
    
    plt.legend(title='Metoda Preprocessingu')
    
    plt.tight_layout()
    plt.savefig('porownanie_resamplingu.png')