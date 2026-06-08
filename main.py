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
from main_visuilizer import defaulVisuilizer, resamplerVisuilizer, featureVisuilizer, comparationVisuilizer, normalizationVisuilizer, testsVizualizations
from sklearn.feature_selection import SelectKBest, f_classif
# from scipy.stats import shapiro, ttest_rel, wilcoxon
from modules.tests import normalizationTest, runTests

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


reduction_names = ["PCA", "SelectKBest"]

# Sekcja 7: Porównanie ilości cech

comparation_results = featureComparasion(X, y, DecisionTreeClassifier(), reduction_names)



np.savez(
    "results/comparation_results.npz", 
    comparation_results=comparation_results,
    reduction_names=reduction_names,
    feature_count = X.shape[1]
)

best_score_index = comparationVisuilizer()

# print(best_score_index)

# Sekcja 7: Esktrakcja vs Selekcja cech

reduction_methods = [
        ("PCA", PCA(n_components=best_score_index+1)),
        ("SelectKBest", SelectKBest(score_func=f_classif, k=best_score_index+1))
    ]

features_results = evaluate_features(X, y, classifiers, column_names, reduction_methods, best_score_index)

np.savez(
    "results/results_features.npz", 
    features_results=features_results, 
    classifiers_names=classifiers_names, 
    reduction_names=reduction_names
)

featureVisuilizer()

# Sekcja 8: Testy

alpha = 0.05

normalization_results = normalizationTest(classifiers_names, reduction_methods, features_results)

np.savez(
    "results/shapiro_results.npz", 
    normalization_results=normalization_results, 
    classifiers_names=classifiers_names, 
    reduction_names=reduction_names,
    alpha = alpha
)

normalizationVisuilizer()

tests_results = runTests(alpha, classifiers_names, reduction_methods, normalization_results, features_results)

np.savez(
    "results/tests_results.npz", 
    tests_results=tests_results, 
    classifiers_names=classifiers_names, 
    reduction_names=reduction_names,
    alpha = alpha
)

testsVizualizations()


# wix_result = wilcoxon(avg_results[:,0], avg_results[:,1])
# print(wix_result)