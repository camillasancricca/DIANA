import random

import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
import feature_selection
from sklearn.preprocessing import OrdinalEncoder





if __name__ == '__main__':
    """
    ds_wine = pd.read_csv('datasets/wine/wine.csv')
    print(ds_wine.head())

    df_list = dirty_compl.dirty(1, "iris", "species", "uniform")
    # print(df_list)
    # ok tutto apposto questo funziona, mi ritorna la lista dei vari datasets con gli errori dentro
    random.shuffle(df_list)
    df_with_missing = df_list[0]
    print(df_with_missing)
    # imputator = data_imputation_tecniques.impute_mean()
    imputator = data_imputation_tecniques.drop()
    imputed_df = imputator.fit_rows(df_with_missing)
    print(imputed_df)
    # tutto ok anche l'imputation va

    score = algorithms_classification.classification(imputed_df, "species", "lr")
    print(score)

    # tutto ok l'algoritmo va, ti ritorna lo score f1 dell'aloritmo

    # ora provo il fatto delle features per tutto il dataset

    names = ["iris", "cancer"]

    with open("data_feature-val.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
                   "p_avg_distinct,p_max_distinct,p_min_distinct," +
                   "avg_density,max_density,min_density," +
                   "avg_entropy,max_entropy,min_entropy," +
                   "p_correlated_features,max_pearson,min_pearson," +
                   "\n")

        for name in names:
            print(name)

            file_path = "datasets/" + name + "/" + name + ".csv"
            df = pd.read_csv(file_path)

            features = str(f.import_features(df))
            features = features.replace("(", "")
            features = features.replace(")", "")
            features = features.replace(" ", "")
            file.write(name + "," + features + "\n")


    # ora provo features singola colonna

    df_name = "iris"
    column = "sepal_length"

    with open("data_feature_single_column.csv", "w") as file:
        file.write("name, column_name, n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
                   "column_uniqueness," + "column_density," + "column_entropy," +
                   "p_correlated_features,max_pearson,min_pearson," + "column_type," +
                   "\n")

        file_path = "datasets/" + df_name + "/" + df_name + ".csv"
        df = pd.read_csv(file_path)

        features = str(f.import_features_single_column(df,column))
        features = features.replace("(", "")
        features = features.replace(")", "")
        features = features.replace(" ", "")
        file.write(df_name + "," + column + "," + features + "\n")


    # prova feature selection

    df_selected = feature_selection.feature_selection(ds_wine, "quality")
    print(df_selected)

    # prova error injection singola colonna

    dirty_compl.dirty_single_column(1, ds_wine, "alcohol", "quality")


    # ora provo features delle singole colonne anche con mean e std

    df_list_new = dirty_compl.dirty(1, "wine", "quality", "uniform")
    df_new = df_list_new[0]
    features = str(f.import_features_single_column2(df_new, "chlorides"))
    features = features.replace("(", "")
    features = features.replace(")", "")
    features = features.replace(" ", "")
    print(features)
    """
    """
    # provo quel problema

    ds_iris = pd.read_csv('datasets/iris/iris.csv')
    print(ds_iris.head())

    df_list = dirty_compl.dirty_single_column(1, ds_iris, "sepal_length", "species")
    # random.shuffle(df_list)
    df_with_missing = df_list[0]
    print(df_with_missing)
    # imputator = data_imputation_tecniques.impute_mean()
    # imputator = data_imputation_tecniques.drop()
    # imputed_df = imputator.fit_rows(df_with_missing)
    imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_mean")
    print(imputed_df)
    # tutto ok anche l'imputation va

    score = algorithms_classification.classification(imputed_df, "species", "lr")
    print(score)

    #provo encoder
    ds_house = pd.read_csv('datasets/house/house.csv')
    print(ds_house)
    """
    """
    columns = ds_car.columns
    cat = list(ds_car.select_dtypes(include=['bool', 'object']).columns)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    oe.fit(ds_car[cat])
    ds_car[cat] = oe.transform(ds_car[cat])
    print(ds_car)
    """
    """
    df_list = dirty_compl.dirty_single_column(1, ds_frogs, "MFCCs_1", "species")
    # random.shuffle(df_list)
    df_with_missing = df_list[0]
    print(df_with_missing)
    imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_linear_regression")
    print(imputed_df)
    """
    """
    ds_car = feature_selection.feature_selection(ds_car,"safety")
    print(ds_car)

    df_list = dirty_compl.dirty_single_column(1, ds_car, "index", "safety")
    # random.shuffle(df_list)
    df_with_missing = df_list[0]
    print(df_with_missing)
    imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_linear_regression")
    print(imputed_df)
    """
    """
    ds_house = feature_selection.feature_selection(ds_house, "SaleCondition")
    print(ds_house)

    df_list = dirty_compl.dirty_single_column(1, ds_house, "SaleType", "SaleCondition")
    # random.shuffle(df_list)
    df_with_missing = df_list[0]
    # features = str(f.import_features_single_column("car", df_with_missing, "SaleType"))
    # print(features)
    imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing.copy(), "impute_logistic_regression")
    print(imputed_df)
    """

    """
    ds_bank = pd.read_csv('datasets/iris/iris.csv')
    print(ds_bank)

    df_list = dirty_compl.dirty(1, "iris", ds_bank.columns[-1], "uniform")
    # print(df_list)
    # random.shuffle(df_list)
    df_with_missing = df_list[0]
    print(df_with_missing)
    imputator = data_imputation_tecniques.impute_linear_and_logistic()
    imputed_df = imputator.fit(df_with_missing, df_with_missing.columns)
    print(imputed_df)
    # tutto ok anche l'imputation va
    """

    ds_wine = pd.read_csv('datasets/bank/bank.csv')
    print(ds_wine.head())

    df_list = dirty_compl.dirty(1, "bank", ds_wine.columns[-1], "uniform")
    df_with_missing = df_list[0]

    imputator = data_imputation_tecniques.no_impute()
    imputed_df = imputator.fit(df_with_missing)
    print(imputed_df)
    score = algorithms_classification.classification(ds_wine, ds_wine.columns[-1], "dt")
    print(score)

















