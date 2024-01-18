import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f

from multiprocessing import Pool
from itertools import repeat
import collections


datasets = ["iris"]
imputation_methods = ["no_impute", "impute_mean", "impute_linear_and_logistic", "impute_mode"]
ML_models = ["dt"]


def generate_seed(n_seed, n_elements):
    seed = []
    seeds = []
    for r in range(0, n_seed):
        for i in range(0, n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds


def parallel_exec_whole(dataset_name, class_name, ML_model):

    # sempre multipli
    n_instances_tot = 2
    n_parallel_jobs = 2
    n_instances_x_job = int(n_instances_tot/n_parallel_jobs)

    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    itr = zip(repeat(dataset_name), repeat(class_name), repeat(ML_model), seed)

    with Pool(processes=n_parallel_jobs) as pool:

        results = pool.starmap(procedure_whole, itr)

        # results quindi è una lista di liste di dictionaries
        # io ora voglio ottenere una singola lista di dictionaries

        # print(results)

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


def procedure_whole(dataset_name, class_name, ML_model, seed):

    df_list = dirty_compl.dirty(seed, dataset_name, class_name, "uniform")

    # questa lista conterrà un dictionary per ogni df in df_list
    list_of_scores = []

    for df_missing in df_list:

        scores = dict()

        for imputation_method in imputation_methods:
            df_missing_copy = df_missing.copy()

            # usando anche la classe per fare imputation
            # imputed_df = data_imputation_tecniques.impute_dataset(df_missing_copy, imputation_method)

            # non usando la classe per fare imputation
            imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_missing_copy, imputation_method, class_name)

            # print(imputation_method)
            ML_score = algorithms_classification.classification(imputed_df, class_name, ML_model)
            scores[imputation_method] = ML_score

        list_of_scores.append(scores)

    return list_of_scores


# MAIN


def main():

    file = open("KB_whole_datasets_three_prova.csv", "w")
    file_scores = open("KB_whole_datasets_three_all_scores.csv", "w")

    file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
               "p_avg_distinct,p_max_distinct,p_min_distinct," +
               "avg_density,max_density,min_density," +
               "avg_entropy,max_entropy,min_entropy," +
               "p_correlated_features,max_pearson,min_pearson," +
               "%missing," + "ML_ALGORITHM," + "BEST_METHOD1," + "SCORE1," +
               "BEST_METHOD2," + "SCORE2," + "BEST_METHOD3," + "SCORE3" +
               "\n")

    for dataset_name in datasets:

        print("-------------" + dataset_name + "-------------")

        dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
        df = pd.read_csv(dataset_path)

        class_name = df.columns[-1]
        feature_list = f.import_feature_whole_pv(dataset_name)  # quindi ho calcolato le features del dataset

        for ML_model in ML_models:

            print("-------------" + ML_model + "-------------")

            scores = parallel_exec_whole(dataset_name, class_name, ML_model)
            print(scores)
            file_scores.write(dataset_name + "," + ML_model + "," + str(scores) + "\n")

            # ora quindi ho una lista di 5 dizionari con tutti gli scores.
            # Ora per ogni dizionario estraggo il miglior metodo
            best_methods = []
            best_methods_scores = []
            second_best_methods = []
            second_best_methods_scores = []
            third_best_methods = []
            third_best_methods_scores = []
            for dictionary in scores:
                best_method = max(dictionary, key=dictionary.get)
                best_method_score = dictionary.get(best_method)
                best_methods.append(best_method)
                best_methods_scores.append(best_method_score)
                # quindi ho salvato il miglior metodo e il suo punteggio
                del dictionary[best_method]
                second_best_method = max(dictionary, key=dictionary.get)
                second_best_method_score = dictionary.get(second_best_method)
                second_best_methods.append(second_best_method)
                second_best_methods_scores.append(second_best_method_score)
                del dictionary[second_best_method]
                third_best_method = max(dictionary, key=dictionary.get)
                third_best_method_score = dictionary.get(third_best_method)
                third_best_methods.append(third_best_method)
                third_best_methods_scores.append(third_best_method_score)

            print("best methods " + str(best_methods))
            print("second best methods " + str(second_best_methods))
            print("third best methods " + str(third_best_methods))

            # ora scrivo le features sul file
            for i in range(0, len(feature_list)):
                file.write(dataset_name + "," + feature_list[i] + "," + ML_model + "," +
                           best_methods[i] + "," + str(best_methods_scores[i]) + "," +
                           second_best_methods[i] + "," + str(second_best_methods_scores[i]) + "," +
                           third_best_methods[i] + "," + str(third_best_methods_scores[i]) + "\n")

    file.close()
    file_scores.close()
    print("end of computation")



if __name__ == '__main__':
    main()







