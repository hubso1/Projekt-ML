import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


def evaluate_resamplers(X, y, classifiers, resamplers):
    """Przeprowadzenie walidację krzyżową na różnych metodach resamplingu.
    Resampling odbywa się wewnątrz pętli, tylko na zbiorze treningowym

    Args:
        X (matrix): Macierz cech
        y (array): Tablica klas
        classifiers (array): tablica klasyfikatorów
        resamplers (array): tablica metod resamplingu
    """

    results = np.zeros((len(classifiers), len(resamplers), 10), dtype=float)


    rsf  = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    for fold_index, (train_index, test_index) in enumerate(rsf.split(X, y)):


        for resampler_index, resampler in enumerate(resamplers):
        
            for clf_index, clf in enumerate(classifiers):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
                clf.fit(X_train_res, y_train_res)
                predict = clf.predict(X_test)
                bac = balanced_accuracy_score(y_test, predict)
                results[clf_index, resampler_index, fold_index] = bac

    return(results)