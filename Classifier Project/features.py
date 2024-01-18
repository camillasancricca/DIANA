import math
import numpy as np
import pandas as pd

import dirty_compl

from operator import add

def density(df):
    den_tot = []

    for attr in df.columns:

        n_distinct = df[attr].nunique()
        prob_attr = []
        den_attr = 0

        for item in df[attr].unique():
            p_attr = len(df[df[attr] == item])/len(df)
            prob_attr.append(p_attr)

        avg_den_attr = 1/n_distinct

        for p in prob_attr:
            den_attr += math.sqrt((p - avg_den_attr) ** 2)
            den_attr = den_attr/n_distinct

        den_tot.append(den_attr*100)

    return den_tot

def entropy(df):
    en_tot = []

    for attr in df.columns:

        prob_attr = []

        for item in df[attr].unique():
            p_attr = len(df[df[attr] == item])/len(df)
            prob_attr.append(p_attr)

        en_attr = 0

        if 0 in prob_attr:
            prob_attr.remove(0)

        for p in prob_attr:
            en_attr += p*np.log(p)
        en_attr = -en_attr

        en_tot.append(en_attr)

    return en_tot

def correlations(df):
    p_corr = 0

    num = list(df.select_dtypes(include=['int64','float64']).columns)

    corr = df[num].corr()

    if len(num) != 0:
        for c in corr.columns:
            a = (corr[c] > 0.7).sum() - 1
            if a > 0:
                p_corr += 1

        p_corr = p_corr / len(corr.columns)

        return round(p_corr,4),round(corr.replace(1,0).max().max(),4),round(corr.min().min(),4)

    else:
        return np.nan,np.nan,np.nan


def import_features(df):
    # per codice whole dataset versione parallela uso questo
    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4), \
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2]


def import_features_2(df):

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    if (len(df) - len(df.copy().dropna())) == 0:
        missing_rows = 1
    else:
        missing_rows = len(df) - len(df.copy().dropna())

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4),\
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2],\
           round((df.isnull().sum().sum())/(rows*cols),4),round(sum(df.isnull().sum() > 0)/cols,4),round(missing_rows/rows,4)

def import_features_3(df,name):

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    if (len(df) - len(df.copy().dropna())) == 0:
        missing_rows = 1
    else:
        missing_rows = len(df) - len(df.copy().dropna())

    return name,rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4),\
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2],\
           round((df.isnull().sum().sum())/(rows*cols),4),round(sum(df.isnull().sum() > 0)/cols,4),round(missing_rows/rows,4)

def import_data_features(name, df):

    with open(name+"_features.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_distinct,p_max_distinct,p_min_distinct,"+
                   "avg_density,max_density,min_density,"+
                   "avg_entropy,max_entropy,min_entropy,"+
                   "p_correlated_features,max_pearson,min_pearson"+
                   "\n")

        features = str(import_features(df))
        features = features.replace("(", "")
        features = features.replace(")", "")
        features = features.replace(" ", "")
        file.write(name+","+features+"\n")

    file_path = name+"_features.csv"
    data_features = pd.read_csv(file_path)

    return data_features

def import_data_features_2(name, df):

    with open(name+"_features.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_distinct,p_max_distinct,p_min_distinct,"+
                   "avg_density,max_density,min_density,"+
                   "avg_entropy,max_entropy,min_entropy,"+
                   "p_correlated_features,max_pearson,min_pearson,"+
                    "%missing,%missing_cols,%missing_rows"+
                   "\n")

        features = str(import_features_2(df))
        features = features.replace("(", "")
        features = features.replace(")", "")
        features = features.replace(" ", "")
        file.write(name+","+features+"\n")

    file_path = name+"_features.csv"
    data_features = pd.read_csv(file_path)

    return data_features

def import_data_features_dataframe(name, df):

    features = import_features_3(df,name)
    #features = features.replace("(", "")
    #features = features.replace(")", "")
    #features = features.replace(" ", "")

    data_features = pd.DataFrame(
        [features],columns=["name","n_tuples","n_attributes","p_num_var","p_cat_var","p_duplicates","total_size",
                   "p_avg_distinct","p_max_distinct","p_min_distinct",
                   "avg_density","max_density","min_density",
                   "avg_entropy","max_entropy","min_entropy",
                   "p_correlated_features","max_pearson","min_pearson",
                    "%missing","%missing_cols","%missing_rows"]
    )

    return data_features

"""
if __name__ == '__main__':

    names = ["iris","cancer","users","wine","soybean","mushrooms","adult","anuran","plants"]

    with open("ALTRO/data_feature-val.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_distinct,p_max_distinct,p_min_distinct,"+
                   "avg_density,max_density,min_density,"+
                   "avg_entropy,max_entropy,min_entropy,"+
                   "p_correlated_features,max_pearson,min_pearson,"+
                   "\n")

        for name in names:

            print(name)

            file_path = "datasets_cl/" + name + "/" + name + ".csv"
            df = pd.read_csv(file_path)

            features = str(import_features(df))
            features = features.replace("(", "")
            features = features.replace(")", "")
            features = features.replace(" ", "")
            file.write(name+","+features+"\n")

"""

# Aggiunti da Enrico:

def density_single_column(df, column):

    n_distinct = df[column].nunique()
    prob_attr = []
    den_attr = 0

    for item in df[column].unique():
        p_attr = len(df[df[column] == item])/len(df)
        prob_attr.append(p_attr)

    avg_den_attr = 1/n_distinct

    for p in prob_attr:
        den_attr += math.sqrt((p - avg_den_attr) ** 2)
        den_attr = den_attr/n_distinct

    return den_attr*100


def entropy_single_column(df, column):

    prob_attr = []

    for item in df[column].unique():
        p_attr = len(df[df[column] == item])/len(df)
        prob_attr.append(p_attr)

    en_attr = 0

    if 0 in prob_attr:
        prob_attr.remove(0)

    for p in prob_attr:
        en_attr += p*np.log(p)
    en_attr = -en_attr

    return en_attr


def import_features_single_column(dataset_name, df, column):

    # questa parte serve per vedere i tipi delle colonne

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    original_dataset = pd.read_csv(dataset_path)

    original_dataset = pd.DataFrame(original_dataset, columns=df.columns)

    num = len(list(original_dataset.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(original_dataset.select_dtypes(include=['bool', 'object']).columns))

    rows = df.shape[0]
    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density_single_column(df, column)
    en = entropy_single_column(df, column)
    corr = correlations(df)

    if original_dataset[column].dtype == 'float64' or original_dataset[column].dtype == 'int64':
        column_type = "numerical"
    else:
        column_type = "categorical"

    """
    "name,column_name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
    "column_uniqueness" +
    "column_density," +
    "column_entropy" +
    "p_correlated_features,max_pearson,min_pearson," +
    "column_type" +
    "%missing"
    """

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(df.duplicated().sum() / rows,
                                                                         4), df.memory_usage().sum(), \
        round(df[column].nunique() / rows, 4),\
        round(den, 4), \
        round(en, 4), \
        corr[0], corr[1], corr[2], \
        column_type, \
        round((df[column].isnull().sum()) / rows, 4)


def import_features_single_column2(dataset_name, df, column):

    # in questa funzione si calcola anche mean e std per le colonne numeriche

    # questa parte serve per vedere i tipi delle colonne

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    original_dataset = pd.read_csv(dataset_path)

    original_dataset = pd.DataFrame(original_dataset, columns=df.columns)

    num = len(list(original_dataset.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(original_dataset.select_dtypes(include=['bool', 'object']).columns))

    rows = df.shape[0]
    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density_single_column(df, column)
    en = entropy_single_column(df, column)
    corr = correlations(df)

    if original_dataset[column].dtype == 'float64' or original_dataset[column].dtype == 'int64':
        column_type = "numerical"
    else:
        column_type = "categorical"

    mean = None
    std = None

    if column_type == "numerical":
        mean = round(df[column].mean(), 4)
        std = round(df[column].std(), 4)

    """
    "name,column_name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
    "column_uniqueness" +
    "column_density," +
    "column_entropy" +
    "p_correlated_features,max_pearson,min_pearson," +
    "column_type" + "mean" + "std"
    "%missing"
    """

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(df.duplicated().sum() / rows,
                                                                         4), df.memory_usage().sum(), \
        round(df[column].nunique() / rows, 4),\
        round(den, 4), \
        round(en, 4), \
        corr[0], corr[1], corr[2], \
        column_type, mean, std,\
        round((df[column].isnull().sum()) / rows, 4)


def import_features_single_column3(dataset_name, df, column):

    # versione usata per parallel version

    # questa parte serve per vedere i tipi delle colonne

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    original_dataset = pd.read_csv(dataset_path)

    original_dataset = pd.DataFrame(original_dataset, columns=df.columns)

    num = len(list(original_dataset.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(original_dataset.select_dtypes(include=['bool', 'object']).columns))

    rows = original_dataset.shape[0]
    cols = original_dataset.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density_single_column(original_dataset, column)
    en = entropy_single_column(original_dataset, column)
    corr = correlations(original_dataset)

    if original_dataset[column].dtype == 'float64' or original_dataset[column].dtype == 'int64':
        column_type = "numerical"
    else:
        column_type = "categorical"

    mean = None
    std = None

    if column_type == "numerical":
        mean = round(original_dataset[column].mean(), 4)
        std = round(original_dataset[column].std(), 4)

    """
    "name,column_name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
    "column_uniqueness" +
    "column_density," +
    "column_entropy" +
    "p_correlated_features,max_pearson,min_pearson," +
    "column_type" + "mean" + "std"
    """

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(original_dataset.duplicated().sum() / rows,
                                                                         4), original_dataset.memory_usage().sum(), \
        round(original_dataset[column].nunique() / rows, 4),\
        round(den, 4), \
        round(en, 4), \
        corr[0], corr[1], corr[2], \
        column_type, mean, std


def import_features_complete(df):

    # in questa versione si calcola anche la percentuale di missing values

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4), \
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2],\
           round((df.isnull().sum().sum()) / (rows * cols), 4)


def import_feature_whole_pv(dataset_name):
    # This function is used to write the features in the case of whole datasets, for the parallel version

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    df = pd.read_csv(dataset_path)

    features = str(import_features(df))
    features = features.replace("(", "")
    features = features.replace(")", "")
    features = features.replace(" ", "")

    # vedo le percentuali precise di missings

    n_iteration = 4  # questo setta il numero di iterazioni da fare per calcolare i missing values
    missing_percentage_list = [0] * 5
    for i in range(0, n_iteration):
        df_list = dirty_compl.dirty(i+1, dataset_name, df.columns[-1], "uniform")
        temp_list = []
        for df_missing in df_list:
            rows = df_missing.shape[0]
            cols = df_missing.shape[1]
            # print(round((df.isnull().sum().sum()) / (rows * cols), 4))
            temp_list.append((df_missing.isnull().sum().sum()) / (rows * cols))

        missing_percentage_list = list(map(add, missing_percentage_list, temp_list))

    missing_percentage_list[:] = [round(x / n_iteration, 4) for x in missing_percentage_list]

    # quindi ora dovrei avere questa lista con i 5 valori di missing values

    # ora devo preparare le 5 righe da ritornare
    features_list = []
    for j in range(0, len(missing_percentage_list)):
        string = features + "," + str(missing_percentage_list[j])
        features_list.append(string)

    return features_list


def import_features_single_column_pv(dataset_name, df, column):
    # This function is used to write the features in the case of single columns, for the parallel version

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    original_dataset = pd.read_csv(dataset_path)

    features = str(import_features_single_column3(dataset_name, df, column))
    features = features.replace("(", "")
    features = features.replace(")", "")
    features = features.replace(" ", "")
    features = features.replace("'", "")

    # questo è il dataset originale ma solo con le colonne selezionate dalla feature selection
    original_dataset_selected = pd.DataFrame(original_dataset, columns=df.columns)

    # vedo le percentuali precise di missings

    n_iteration = 4  # questo setta il numero di iterazioni da fare per calcolare i missing values
    missing_percentage_list = [0] * 5
    for i in range(0, n_iteration):
        df_list = dirty_compl.dirty_single_column(i+1, original_dataset_selected, column, original_dataset.columns[-1])
        temp_list = []
        for df_missing in df_list:
            rows = df_missing.shape[0]
            cols = df_missing.shape[1]
            # print(round((df.isnull().sum().sum()) / (rows * cols), 4))
            temp_list.append((df_missing[column].isnull().sum()) / rows)

        missing_percentage_list = list(map(add, missing_percentage_list, temp_list))

    missing_percentage_list[:] = [round(x / n_iteration, 4) for x in missing_percentage_list]

    # quindi ora ho questa lista con i 5 valori di missing values

    # ora devo preparare le 5 righe da ritornare
    features_list = []
    for j in range(0, len(missing_percentage_list)):
        string = features + "," + str(missing_percentage_list[j])
        features_list.append(string)

    return features_list




















