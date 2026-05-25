from tabulate import tabulate
from modules.visualisation import saveResamplingBarChart
import numpy as np

def fileImport():
    data_basic = np.load("results/results_basic.npz", allow_pickle=True)
    results = data_basic["results"]
    classifiers_names = data_basic["classifiers_names"].tolist()
    data_resampling = np.load("results/results_resampling.npz", allow_pickle=True)
    resampling_results = data_resampling["resampling_results"]
    resamplers_names = data_resampling["resamplers_names"].tolist()

    return results, classifiers_names, resampling_results, resamplers_names


def resamplerVisuilizer():
    results, classifiers_names, resampling_results, resamplers_names = fileImport()

    console_output_array = []

    for classifier_index, classifier in enumerate(classifiers_names):

        row = [classifier]

        for resampler_index, resampler in enumerate(resamplers_names):
            mean = np.mean(resampling_results[classifier_index, resampler_index])
            std = np.std(resampling_results[classifier_index, resampler_index])

            result_str = f"{mean:.3f} ± {std:.3f}"
            row.append(result_str)

        console_output_array.append(row)
    

    print(tabulate(console_output_array, headers=[""] + resamplers_names, tablefmt="fancy_grid"))


    saveResamplingBarChart(results, resampling_results, classifiers_names, resamplers_names)

def defaulVisuilizer():
    results, classifiers_names, resampling_results, resamplers_names = fileImport()
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

def featureVisuilizer():
    try:
        data_features = np.load("results/results_features.npz", allow_pickle=True)
        features_results = data_features["features_results"]
        classifiers_names = data_features["classifiers_names"].tolist()
        reduction_names = data_features["reduction_names"].tolist()
    except FileNotFoundError:
        print("\n Nie znaleziono pliku results_features.npz")
        return

    console_output_array = []

    for classifier_index, classifier in enumerate(classifiers_names):
        row = [classifier] 


        for red_index, red_name in enumerate(reduction_names):

            mean = np.mean(features_results[classifier_index, red_index])
            std = np.std(features_results[classifier_index, red_index])

            result_str = f"{mean:.3f} ± {std:.3f}"
            row.append(result_str)

        console_output_array.append(row)

    print(tabulate(console_output_array, headers=["Klasyfikator"] + reduction_names, tablefmt="fancy_grid"))