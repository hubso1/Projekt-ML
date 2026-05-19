import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss



def evaluate_resamplers(X, y, classifiers, classifiers_name):
    """
    Przeprowadzenie walidację krzyżową na różnych metodach resamplingu.
    Resampling odbywa się wewnątrz pętli, tylko na zbiorze treningowym
    """
    resamplers = [
        ("SMOTE", SMOTE()),
        ("BorderlineSMOTE", BorderlineSMOTE()),
        ("RUS", RandomUnderSampler()),
        ("NearMiss", NearMiss())
    ]

    results = {}

    rsf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)

    for name, resampler in resamplers:

        for index, clf in enumerate(classifiers):

            clf_name = classifiers_name[index]

            bac_scores = []

            for train_index, test_index in rsf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]


                X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)

                clf.fit(X_train_res, y_train_res)
                predict = clf.predict(X_test)

                bac = balanced_accuracy_score(y_test, predict)
                bac_scores.append(bac)

            key = f"{clf_name} + {name}"
            results[key] = bac_scores
        
    return results