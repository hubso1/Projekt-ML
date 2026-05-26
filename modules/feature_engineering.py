import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE

def evaluate_features(X, y, classifiers, column_names):
    """
    Przeprowadza walidację krzyżową łączącą zbalansowanie (BorderlineSMOTE), 
    standaryzację danych oraz redukcję wymiarowości do 2 cech.
    """

    reduction_methods = [
        ("PCA", PCA(n_components=2)),
        ("SelectKBest", SelectKBest(score_func=f_classif, k=2))
    ]

    reduction_names = ["PCA", "SelectKBest"]

    results = np.zeros((len(classifiers), len(reduction_methods), 10), dtype=float)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    smote = BorderlineSMOTE()
    scaler = StandardScaler()

    for fold_index, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        

        for reduction_index, (name, reducer) in enumerate(reduction_methods):

            if name == "SelectKBest":
                X_train_reduced = reducer.fit_transform(X_train_res, y_train_res)

                if fold_index == 0:
                    selected_mask = reducer.get_support()
                    best_features = np.array(column_names[:-1])[selected_mask]
                    print(f"\n[INFO] SelectKBest wybrało 2 najważniejsze cechy: {best_features}")
            
            else:
                X_train_reduced = reducer.fit_transform(X_train_res)

            X_test_reduced= reducer.transform(X_test_scaled)

            for clf_index, clf in enumerate(classifiers):

                clf.fit(X_train_reduced, y_train_res)
                predict = clf.predict(X_test_reduced)
                    
                bac = balanced_accuracy_score(y_test, predict)
                results[clf_index, reduction_index, fold_index] = bac

    return results, reduction_names
