import random

import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
import feature_selection
from sklearn.preprocessing import OrdinalEncoder
import script_parallelismo
from operator import add



if __name__ == '__main__':

    # ds_iris = pd.read_csv('datasets/iris/iris.csv')
    # print(ds_iris)

    """
    df_list = dirty_compl.dirty(1, "iris", "species", "uniform")
    # print(df_list)
    # random.shuffle(df_list)

    df_with_missing = df_list[0]
    print(df_with_missing)
    imputator = data_imputation_tecniques.impute_mean()
    imputed_df = imputator.fit_mode(df_with_missing)
    print(imputed_df)

    # score = algorithms_classification.classification(imputed_df, "species", "lr")
    # print(score)

    """

    # results = script_parallelismo.parallel_exec_enrico("iris", "species", "lr")
    # print(results)

    """
    ds_iris = pd.read_csv('datasets/iris/iris.csv')

    ds_iris2 = feature_selection.feature_selection(ds_iris, "species")

    n_iteration = 4  # questo setta il numero di iterazioni da fare per calcolare i missing values
    missing_percentage_list = [0] * 5
    for i in range(0, n_iteration):
        df_list = dirty_compl.dirty_single_column(i + 1, ds_iris2, "sepal_length", ds_iris.columns[-1])
        temp_list = []
        for df in df_list:
            rows = df.shape[0]
            cols = df.shape[1]
            # print(round((df.isnull().sum().sum()) / (rows * cols), 4))
            temp_list.append((df["sepal_length"].isnull().sum()) / rows)
        print(temp_list)
        missing_percentage_list = list(map(add, missing_percentage_list, temp_list))

    missing_percentage_list[:] = [round(x / n_iteration, 4) for x in missing_percentage_list]

    print(df_list)
    print(missing_percentage_list)
    """

    """
    original_dataset = pd.read_csv('datasets/iris/iris.csv')
    ds_iris2 = feature_selection.feature_selection(original_dataset, "species")


    # questo è il dataset originale ma solo con le colonne selezionate dalla feature selection
    original_dataset_selected = pd.DataFrame(original_dataset, columns=ds_iris2.columns)

    # vedo le percentuali precise di missings

    n_iteration = 4  # questo setta il numero di iterazioni da fare per calcolare i missing values
    missing_percentage_list = [0] * 5
    for i in range(0, n_iteration):
        df_list = dirty_compl.dirty_single_column(i+1, original_dataset_selected, "sepal_length", original_dataset.columns[-1])
        temp_list = []
        for df in df_list:
            rows = df.shape[0]
            cols = df.shape[1]
            # print(round((df.isnull().sum().sum()) / (rows * cols), 4))
            temp_list.append((df["sepal_length"].isnull().sum()) / rows)

        missing_percentage_list = list(map(add, missing_percentage_list, temp_list))

    missing_percentage_list[:] = [round(x / n_iteration, 4) for x in missing_percentage_list]

    print(missing_percentage_list)
    """

    """
    original_dataset = pd.read_csv('datasets/wine/wine.csv')
    ds_iris2 = feature_selection.feature_selection(original_dataset, "quality")

    features = f.import_features_single_column_pv("wine", ds_iris2, "alcohol")
    print(features)
    """
    original_dataset = pd.read_csv('datasets/wine/wine.csv')
    for i in range(0, 4):
        df_list = dirty_compl.dirty_single_column(i+1, original_dataset, "alcohol", "quality")
        df_missing = df_list[0]
        missing = df_missing["alcohol"].isnull().sum() / df_missing.shape[0]
        print(missing)









