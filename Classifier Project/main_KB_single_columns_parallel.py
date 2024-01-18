import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
import feature_selection

from multiprocessing import Pool
from itertools import repeat
import collections


datasets = ["cancer", "iris"]
imputation_methods = ["no_impute", "impute_mean",
                      "impute_logistic_regression"]
ML_models = ["dt", "lr"]


def generate_seed(n_seed, n_elements):
    seed = []
    seeds = []
    for r in range(0, n_seed):
        for i in range(0, n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds


def parallel_exec_single(df, dataset_name, class_name, column, ML_model):

    # sempre multipli
    n_instances_tot = 2
    n_parallel_jobs = 2
    n_instances_x_job = int(n_instances_tot/n_parallel_jobs)

    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    itr = zip(repeat(df), repeat(dataset_name), repeat(class_name), repeat(column), repeat(ML_model), seed)

    with Pool(processes=n_parallel_jobs) as pool:

        results = pool.starmap(procedure_single, itr)

        # results quindi è una lista di liste di dictionaries
        # io ora voglio ottenere una singola lista di dictionaries

        print(results)

        # ora aggrego i risultati ottenuti dai vari processi paralleli
        results_final = []
        # codice da controllare
        for i in range(0, len(results[0])):
            # creo lista di corrispondenti dizionari
            dictionaries_list = []
            for original_list in results:
                dictionaries_list.append(original_list[i])
            # print(dictionaries_list)

            # ora aggrego questi dizionari
            counter = collections.Counter()
            for d in dictionaries_list:
                counter.update(d)
            aggregated_dictionary = dict(counter)
            # ora ho che il dizionario è la somma dei dizionari, ora devo dividere (per fare la media)
            for key in aggregated_dictionary:
                aggregated_dictionary[key] = aggregated_dictionary[key]/len(results)
            # quindi ora dovrei avere dizionario che è la media dei corrispondenti dizionari
            # print(aggregated_dictionary)
            results_final.append(aggregated_dictionary)

        return results_final


def procedure_single(df, dataset_name, class_name, column, ML_model, seed):

    df_list = dirty_compl.dirty_single_column(seed, df, column, class_name)

    # questa lista conterrà un dictionary per ogni df in df_list
    list_of_scores = []

    for df_missing in df_list:

        scores = dict()

        for imputation_method in imputation_methods:
            if not data_imputation_tecniques.check_technique_compatibility(dataset_name, imputation_method, column):
                continue
            df_missing_copy = df_missing.copy()

            # usando anche la classe per fare imputation
            # imputed_df = data_imputation_tecniques.impute_dataset(df_missing_copy, imputation_method)

            # non usando la classe per fare imputation
            imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_missing_copy, imputation_method,
                                                                           class_name)
            # print(imputation_method)
            ML_score = algorithms_classification.classification(imputed_df, class_name, ML_model)
            scores[imputation_method] = ML_score

        list_of_scores.append(scores)

    return list_of_scores


# MAIN

def main():

    file = open("KB_single_columns_prova.csv", "w")

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

        df = feature_selection.feature_selection(df, class_name)

        columns = list(df.columns)
        columns.remove(class_name)

        for column in columns:

            print("-------------" + column + "-------------")

            feature_list = f.import_features_single_column_pv(dataset_name, df, column)
            # quindi ho calcolato le features della colonna

            for ML_model in ML_models:

                print("-------------" + ML_model + "-------------")

                scores = parallel_exec_single(df, dataset_name, class_name, column, ML_model)
                print(scores)
                # ora quindi ho una lista di 5 dizionari con tutti gli scores.
                # Ora per ogni dizionario estraggo il miglior metodo

                best_methods = []
                for dictionary in scores:
                    best_method = max(dictionary, key=dictionary.get)
                    best_methods.append(best_method)
                print("best methods " + str(best_methods))

                # ora scrivo le features sul file
                for i in range(0, len(feature_list)):
                    file.write(dataset_name + "," + column + "," + feature_list[i] + "," + ML_model + "," + best_methods[i] + "\n")

    file.close()
    print("end of computation")


if __name__ == '__main__':
    main()













