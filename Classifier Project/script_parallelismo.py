import collections

import pandas as pd

import dirty_compl as dc
import features as f
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import algorithms_classification as ac
# import imputation_tecniques as it
import best_method as bm
from multiprocessing import Pool
from itertools import repeat

import algorithms_classification
import dirty_compl
import data_imputation_tecniques


def generate_seed(n_seed,n_elements):
    seed = []
    seeds = []
    for r in range(0,n_seed):
        for i in range(0,n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds


def parallel_exec(name, class_name):

    # sempre multipli
    n_instances_tot = 16
    n_parallel_jobs = 8
    n_instances_x_job = int(n_instances_tot/n_parallel_jobs)

    #decision_trees = 0
    #knn = 0
    #naive_bayes = 0
    #logistic_r = 0

    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    itr = zip(repeat(name), repeat(class_name), seed)

    with Pool(processes=n_parallel_jobs) as pool:

        results = pool.starmap(procedure, itr)

        print(results)

        #results_final = np.array([decision_trees,knn,naive_bayes,logistic_r])

        for instance in range(0, n_parallel_jobs):
            results_final = results_final + results[instance]

        results_final = results_final/n_parallel_jobs
        print(results_final)
        return results_final


def procedure(name, class_name, seed):
    print("il vostro main")



if __name__ == '__main__':

    name = "stars"
    class_name = "Type"
    #parametri che vi servono in ingresso alla procedure

    [acc_dt,acc_knn,acc_nb,acc_lr] = parallel_exec(name, class_name) #prende results final per scriverlo su csv

    #print("ACCURACY DT: "+ str(acc_dt))
    #print("ACCURACY KNN: " + str(acc_knn))
    #print("ACCURACY NB: " + str(acc_nb))
    #print("ACCURACY LR: " + str(acc_lr))

    with open(name + "_validation.csv", "w") as file:
        file.write("NAME,ML_ALGORITHM,ACCURACY\n")

        file.write(name + ",DT," + str(acc_dt) + "\n")
        file.write(name + ",KNN," + str(acc_knn) + "\n")
        file.write(name + ",NB," + str(acc_nb) + "\n")
        file.write(name + ",LR," + str(acc_lr) + "\n")


# codice aggiunto da Enrico

def parallel_exec_enrico(dataset_name, class_name, ML_model):

    # sempre multipli
    n_instances_tot = 2
    n_parallel_jobs = 2
    n_instances_x_job = int(n_instances_tot/n_parallel_jobs)

    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    itr = zip(repeat(dataset_name), repeat(class_name), repeat(ML_model), seed)

    with Pool(processes=n_parallel_jobs) as pool:

        results = pool.starmap(procedure_enrico, itr)

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


def procedure_enrico(dataset_name, class_name, ML_model, seed):

    imputation_methods = ["no_impute", "impute_mean", "impute_linear_and_logistic"]  # questo poi vedo dove metterlo

    df_list = dirty_compl.dirty(seed, dataset_name, class_name, "uniform")

    # questa lista conterrà un dictionary per ogni df in df_list
    list_of_scores = []

    for df_missing in df_list:

        scores = dict()

        for imputation_method in imputation_methods:
            df_missing_copy = df_missing.copy()
            imputed_df = data_imputation_tecniques.impute_dataset(df_missing_copy, imputation_method)
            # print(imputation_method)
            ML_score = algorithms_classification.classification(imputed_df, class_name, ML_model)
            scores[imputation_method] = ML_score

        list_of_scores.append(scores)

    return list_of_scores


















