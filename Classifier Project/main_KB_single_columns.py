import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
import feature_selection




if __name__ == '__main__':

    datasets = ["users", "mushrooms", "german"]
    imputation_methods = ["no_impute", "impute_mean", "impute_bfill", "impute_linear_regression", "impute_logistic_regression"]
    ML_models = ["nb", "knn"]

    # print()

    file = open("KB_single_columns_new_datasets.csv", "w")

    file.write("name, column_name, n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
               "column_uniqueness," + "column_density," + "column_entropy," +
               "p_correlated_features,max_pearson,min_pearson," + "column_type," + "mean," + "std," +
               "%missing," + "ML_ALGORITHM," + "BEST_METHOD" +
               "\n")

    for dataset_name in datasets:

        print("-------------" + dataset_name + "-------------")

        dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
        df = pd.read_csv(dataset_path)

        class_name = df.columns[-1]

        # qua faccio feature selection
        df = feature_selection.feature_selection(df, class_name)

        columns = list(df.columns)
        columns.remove(class_name)
        # print(columns)
        # print(class_name)

        for column in columns:
            # considero ogni colonna del dataset singolarmente

            print("-------------" + column + "-------------")

            df_list = dirty_compl.dirty_single_column(1, df, column, class_name)
            # print(df_list)

            for ML_model in ML_models:

                print("-------------" + ML_model + "-------------")

                for df_missing in df_list:

                    scores = dict()

                    for imputation_method in imputation_methods:
                        if not data_imputation_tecniques.check_technique_compatibility(dataset_name, imputation_method, column):
                            continue
                        df_missing_copy = df_missing.copy()
                        imputed_df = data_imputation_tecniques.impute_dataset(df_missing_copy, imputation_method)
                        # qua devo ora vedere le performance
                        print(imputation_method)
                        ML_score = algorithms_classification.classification(imputed_df, class_name, ML_model)
                        scores[imputation_method] = ML_score
                        print(ML_score)

                    best_method = max(scores, key=scores.get)
                    print("best method " + best_method)

                    features = str(f.import_features_single_column2(dataset_name, df_missing, column))
                    features = features.replace("(", "")
                    features = features.replace(")", "")
                    features = features.replace(" ", "")
                    features = features.replace("'", "")
                    file.write(dataset_name + "," + column + "," + features + "," + ML_model + "," + best_method + "\n")

    file.close()










