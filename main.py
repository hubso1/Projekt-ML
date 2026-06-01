import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modules.visualisation import saveHeatMap, checkData, checkClassBinalse, saveResamplingBarChart
from modules.pattern_recognition import basicPatternRecognition
from modules.resampling import evaluate_resamplers
from modules.feature_engineering import evaluate_features, featureComparasion
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tabulate import tabulate
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from main_visuilizer import defaulVisuilizer, resamplerVisuilizer, featureVisuilizer, comparationVisuilizer
from sklearn.feature_selection import SelectKBest, f_classif

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

np.savez(
    "results/results_basic.npz", 
    results=results, 
    classifiers_names=classifiers_names
)


# Sekcja 6: Over/Under-Sampling

resamplers = [SMOTE(), BorderlineSMOTE(), RandomUnderSampler(), NearMiss()]
resamplers_names = ["SMOTE", "BorderlineSMOTE", "RUS", "NearMiss"]


resampling_results = evaluate_resamplers(X, y, classifiers, resamplers)

np.savez(
    "results/results_resampling.npz", 
    resampling_results=resampling_results, 
    classifiers_names=classifiers_names, 
    resamplers_names=resamplers_names
)

defaulVisuilizer()
resamplerVisuilizer()

# print(resampling_results)
reduction_methods = [
        ("PCA", PCA(n_components=2)),
        ("SelectKBest", SelectKBest(score_func=f_classif, k=2))
    ]

reduction_names = ["PCA", "SelectKBest"]

# Sekcja 7: Porównanie ilości cech

comparation_results = featureComparasion(X, y, DecisionTreeClassifier(), reduction_names)

# Sekcja 7: Esktrakcja vs Selekcja cech

np.savez(
    "results/comparation_results.npz", 
    comparation_results=comparation_results,
    reduction_names=reduction_names,
    feature_count = X.shape[1]
)

comparationVisuilizer()


# features_results = evaluate_features(X, y, classifiers, column_names, reduction_methods)

# np.savez(
#     "results/results_features.npz", 
#     features_results=features_results, 
#     classifiers_names=classifiers_names, 
#     reduction_names=reduction_names
# )

# featureVisuilizer()