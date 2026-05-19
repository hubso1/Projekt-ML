import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from modules.visualisation import saveHeatMap, checkData, checkClassBinalse, saveResamplingBarChart
from modules.pattern_recognition import basicPatternRecognition
from modules.resampling import evaluate_resamplers
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tabulate import tabulate

# Sekcja 1: Wczytanie danych
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


# Sekcja 2: Sprawdzenie zbalansowania klas

checkClassBinalse(y)

# Sekcja 3: Sprawdzenie danych

checkData(X)

# Sekcja 4: Wizualizacja korelacji cech

saveHeatMap(X, column_names)

# Sekcja 5: Trenowanie i testowanie

classifiers = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC()]
classifiers_names = ["GNB", "KNN", "DTC", "SVC"]

results = basicPatternRecognition(X, y, classifiers)

console_output_array = []

for index, model_name in enumerate(classifiers_names):
    mean = np.mean(results[index])
    std = np.std(results[index])
    result_str = f"{mean:.3f} ± {std:.3f}"

    if mean >= 0.99:
        meaning = "Dane syntetyczne"
    else:
        meaning = "Dane nie syntetyczne"

    console_output_array.append([model_name, result_str, meaning])

print(tabulate(console_output_array, headers=["Model", "Wyniki (BAC)", "Charakter etykiety"], tablefmt="fancy_grid"))


# Sekcja 6: Over/Under-Sampling
resampling_results = evaluate_resamplers(X, y, classifiers, classifiers_names)

table_to_display = []
for combination_name, scores in resampling_results.items():
    mean = np.mean(scores)
    std = np.std(scores)
    result_str = f"{mean:.3f} ± {std:.3f}"

    table_to_display.append([combination_name, result_str])

print(tabulate(table_to_display, headers=["Klasyfikator + Metoda", "Średni wynik (BAC)"], tablefmt="fancy_grid"))

saveResamplingBarChart(results, resampling_results, classifiers_names)