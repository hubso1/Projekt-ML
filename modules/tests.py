from scipy.stats import shapiro, ttest_rel, wilcoxon
import numpy as np

def normalizationTest(classifiers_names, reduction_methods, features_results):

    results = np.zeros((len(classifiers_names), len(reduction_methods), 2), dtype=float)

    for classfier_index, classfier in enumerate(classifiers_names):
        for reduction_index, reduction in enumerate(reduction_methods):
            shapiro_results = shapiro(features_results[classfier_index, reduction_index])
            # print(shapiro_results)
            results[classfier_index, reduction_index, 0] = shapiro_results[0]
            results[classfier_index, reduction_index, 1] = shapiro_results[1]

    return results



def run_tests(alpha, classifiers_names, reduction_methods, shapiro_results):
    for classifier_index, classifier in enumerate(classifiers_names):
        for reduction_index, reduxtion_name in enumerate(reduction_methods):
            if shapiro_results[classifier_index, reduction_index, 1] > alpha:
                result = test()
            else:
                result = wilcoxon()
            
            print(result)



def ttest():
    pass

def wilcoxon():
    pass