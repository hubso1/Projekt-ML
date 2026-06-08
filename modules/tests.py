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



def runTests(alpha, classifiers_names, reduction_methods, shapiro_results, features_results):

    results = np.zeros((len(classifiers_names), 2), dtype=float)
    name_pca = reduction_methods[0][0]
    name_kselect = reduction_methods[1][0]

    for classifier_index, classifier in enumerate(classifiers_names):

        p_val_pca = shapiro_results[classifier_index, 0, 1]
        p_val_kselect = shapiro_results[classifier_index, 1, 1]
        
        data_pca = features_results[classifier_index, 0]
        data_kselect = features_results[classifier_index, 1]

        if p_val_pca > alpha and p_val_kselect > alpha:
            stat, p = runTtest(data_pca, data_kselect)
            
        else:
            stat, p = runWixicon(data_pca, data_kselect)
            
        results[classifier_index, 0] = stat
        results[classifier_index, 1] = p

        return results
        
        

        



def runTtest(data1, data2):
    stat, p_value = ttest_rel(data1, data2)
    return stat, p_value

def runWixicon(data1, data2):
    stat, p_value = wilcoxon(data1, data2)
    return stat, p_value