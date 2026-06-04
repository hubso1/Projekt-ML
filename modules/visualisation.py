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


def saveResamplingBarChart(baseline_results, resampling_results, classifiers_names, resampling_names):
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

    for classifier_index, classifier_result in enumerate(resampling_results):
        for resampler_index, resampler_result in enumerate (classifier_result):
            all_results.append({'Klasyfikator': classifiers_names[classifier_index], 'Metoda': resampling_names[resampler_index], 'BAC': np.mean(resampler_result)})

    df = pd.DataFrame(all_results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Klasyfikator', y='BAC', hue='Metoda', palette='viridis')

    plt.title('Porównanie skuteczności klasyfikatorów z różnymi metodami resamplingu')
    plt.ylabel('Balanced Accuracy (BAC)')
    plt.ylim(0, 1.1)
    
    plt.legend(title='Metoda Preprocessingu')
    
    plt.tight_layout()
    plt.savefig('porownanie_resamplingu.png')


def saveExtractionChart(features_results, classifiers_names, reduction_names):
    """ Generowanie grupowego wykresu słupkowego porównujący wyniki ekstrakcji cech.

    Args:
        features_results (array): Wyniki ekstrakcji.
        reduction_names (list): Nazwy testowanych funkcji redukcji cech.
        classifiers_names (list): Nazwy testowanych klasyfikatorów.
    """
    all_results = []


    for classifier_index, classifier_result in enumerate(features_results):
        for resampler_index, resampler_result in enumerate (classifier_result):
            all_results.append({'Klasyfikator': classifiers_names[classifier_index], 'Metoda': reduction_names[resampler_index], 'BAC': np.mean(resampler_result)})

    df = pd.DataFrame(all_results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Klasyfikator', y='BAC', hue='Metoda', palette='viridis')

    plt.title('Porównanie skuteczności metod redukcji cech')
    plt.ylabel('Balanced Accuracy (BAC)')
    plt.ylim(0, 1.1)
    
    plt.legend(title='Metoda Preprocessingu')
    
    plt.tight_layout()
    plt.savefig('porownanie_ekstrakcji.png')

def saveFeatureCountComparasionChart(comparation_results, reduction_names, feature_count):
    """ Generowanie wykresu liniowego pokazującego BAC w zależności od liczby cech.

    Args:
        comparation_results (array): Wyniki porównania (Metoda x Liczba cech x Folds)
        reduction_names (list): Nazwy metod (PCA, SelectKBest)
        feature_count (int): Maksymalna liczba cech
    """

    plt.figure(figsize=(12, 7))
    
    x_values = np.arange(1, feature_count + 1)
    
    colors = ['#1f77b4', '#ff7f0e'] 
    
    for reduction_index, name in enumerate(reduction_names):
        means = np.mean(comparation_results[reduction_index], axis=1)
        
        plt.plot(x_values, means, marker='o', label=name, linewidth=2.5, markersize=8, color=colors[reduction_index])
        
        # Wartosci przy kropkach na wykresie
        for i, txt in enumerate(means):
            y_offset = 0.005 if reduction_index == 1 else -0.015
            va_align = 'bottom' if reduction_index == 1 else 'top'
            
            plt.text(x_values[i], txt + y_offset, f'{txt:.3f}', 
                     ha='center', va=va_align, fontsize=9, fontweight='bold', color=colors[reduction_index])
        
    plt.title('Wpływ liczby cech na skuteczność modelu (Decision Tree Classifier)')
    plt.xlabel('Liczba cech (k / n_components)')
    plt.ylabel('Balanced Accuracy (BAC)')
    
    plt.xticks(x_values)
    
    # Dynamiczne skalowanie zakresu osi Y
    min_val = np.min(comparation_results)
    max_val = np.max(comparation_results)
    plt.ylim(min_val - 0.05, max_val + 0.05) 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Metoda Redukcji')
    
    plt.tight_layout()
    plt.savefig('porownanie_ilosci_cech.png')