import random

import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f




if __name__ == '__main__':

    datasets = ["car", "cancer", "wine"]
    imputation_methods = ["no_impute", "impute_mean", "impute_linear_and_logistic", "impute_knn", "impute_mice", "impute_soft"]
    ML_models = ["dt", "lr"]

    file = open("KB_whole_datasets.csv", "w")

    file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
               "p_avg_distinct,p_max_distinct,p_min_distinct," +
               "avg_density,max_density,min_density," +
               "avg_entropy,max_entropy,min_entropy," +
               "p_correlated_features,max_pearson,min_pearson," +
               "%missing," + "ML_ALGORITHM," + "BEST_METHOD" +
               "\n")

    for dataset_name in datasets:

        print("-------------" + dataset_name + "-------------")

        dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
        df = pd.read_csv(dataset_path)

        class_name = df.columns[-1]

        df_list = dirty_compl.dirty(1, dataset_name, class_name, "uniform")

        for ML_model in ML_models:

            print("-------------" + ML_model + "-------------")

            for df_missing in df_list:

                scores = dict()

                for imputation_method in imputation_methods:
                    df_missing_copy = df_missing.copy()
                    imputed_df = data_imputation_tecniques.impute_dataset(df_missing_copy, imputation_method)
                    # qua devo ora vedere le performance
                    print(imputation_method)
                    ML_score = algorithms_classification.classification(imputed_df, class_name, ML_model)
                    scores[imputation_method] = ML_score
                    print(ML_score)

                best_method = max(scores, key=scores.get)
                print("best method " + best_method)

                features = str(f.import_features_complete(df_missing))
                features = features.replace("(", "")
                features = features.replace(")", "")
                features = features.replace(" ", "")
                file.write(dataset_name + "," + features + "," + ML_model + "," + best_method + "\n")

    file.close()

















