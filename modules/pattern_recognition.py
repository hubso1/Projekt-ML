import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold


def basicPatternRecognition(X, y, classifiers):
    """ Trenowanie i testowanie wybranych klasyfikatorów 

    Args:
        X (matrix): Macierz cech
        y (array): Tablica klas
        classifiers (array): tablica klasyfikatorów
    """
    results = np.zeros((len(classifiers), 10), dtype=float)
    rsf  = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
    for i, (train_index, test_index) in enumerate(rsf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train, y_train)
            predict = classifier.predict(X_test)
            bac = balanced_accuracy_score(y_test, predict)
            results[index, i] = bac
    return(results)

def checkSyntecicLabel(result):
    pass