import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE

def evaluate_features(X, y, classifiers, column_names, reduction_methods):
    """
    Przeprowadza walidację krzyżową łączącą zbalansowanie (BorderlineSMOTE), 
    standaryzację danych oraz redukcję wymiarowości do 2 cech.
    """

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

    return results


def featureComparasion(X, y, classifier, reduction_names):
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2)
    smote = BorderlineSMOTE()
    scaler = StandardScaler()

    max_features = X.shape[1]

    results = np.zeros(( len(reduction_names), max_features, 10), dtype=float)

    for fold_index, (train_index, test_index) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        for feature_index in range(0, max_features):
            
        

            for reduction_index, name in enumerate(reduction_names):

                if name == "SelectKBest":
                    reducer = SelectKBest(score_func=f_classif, k=feature_index+1)
                    X_train_reduced = reducer.fit_transform(X_train_res, y_train_res)
                
                else:
                    reducer = PCA(n_components=feature_index+1)
                    X_train_reduced = reducer.fit_transform(X_train_res)

                X_test_reduced= reducer.transform(X_test_scaled)

                classifier.fit(X_train_reduced, y_train_res)
                predict = classifier.predict(X_test_reduced)
                        
                bac = balanced_accuracy_score(y_test, predict)
                results[reduction_index, feature_index, fold_index] = bac

    return results