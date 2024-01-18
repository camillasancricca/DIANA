import math
import numpy as np
import pandas as pd

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


def import_features_2(df):

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4),\
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2]


def import_data_features(name, df):

    with open(name+"_features.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_distinct,p_max_distinct,p_min_distinct,"+
                   "avg_density,max_density,min_density,"+
                   "avg_entropy,max_entropy,min_entropy,"+
                   "p_correlated_features,max_pearson,min_pearson"+
                   "\n")

        features = str(import_features_2(df))
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

    features = import_features_2(df,name)
    #features = features.replace("(", "")
    #features = features.replace(")", "")
    #features = features.replace(" ", "")

    data_features = pd.DataFrame(
        [features],columns=["name","n_tuples","n_attributes","p_num_var","p_cat_var","p_duplicates","total_size",
                   "p_avg_distinct","p_max_distinct","p_min_distinct",
                   "avg_density","max_density","min_density",
                   "avg_entropy","max_entropy","min_entropy",
                   "p_correlated_features","max_pearson","min_pearson"]
    )

    return data_features


if __name__ == '__main__':

    names = ["acusticFS","cancerFS","ecoliFS","frogsFS","letterFS","oilFS","qualityredFS","qualitywhiteFS"]

    with open("/Users/martinacaffagnini/Tesi/CodiceTesi/datasets/features_datasets.csv", "w") as file:
        file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_distinct,p_max_distinct,p_min_distinct,"+
                   "avg_density,max_density,min_density,"+
                   "avg_entropy,max_entropy,min_entropy,"+
                   "p_correlated_features,max_pearson,min_pearson,"+
                   "\n")

        for name in names:

            print(name)
            file_path = "/Users/martinacaffagnini/Tesi/CodiceTesi/datasets/" + name+ ".csv"
            df = pd.read_csv(file_path)

            features = str(import_features_2(df))
            features = features.replace("(", "")
            features = features.replace(")", "")
            features = features.replace(" ", "")
            file.write(name+","+features+"\n")

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


def import_features_single_column(df, column):
    num = len(list(df.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool', 'object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density_single_column(df, column)
    en = entropy_single_column(df, column)
    corr = correlations(df)

    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        column_type = "numerical"
    else:
        column_type = "categorical"

    """
    "name,column_name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
    "column_uniqueness" +
    "column_density," +
    "column_entropy" +
    "p_correlated_features,max_pearson,min_pearson," +
    "column_type"    """

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(df.duplicated().sum() / rows,
                                                                         4), df.memory_usage().sum(), \
        round(df[column].nunique() / rows, 4),\
        round(den, 4), \
        round(en, 4), \
        corr[0], corr[1], corr[2], \
        column_type, \
        round((df[column].isnull().sum()) / rows, 4)  # da controllare



def import_features_complete(df):


    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4), \
           round(sum(den)/len(den),4),round(max(den),4),round(min(den),4),\
           round(sum(en)/len(en),4),round(max(en),4),round(min(en),4),\
           corr[0],corr[1],corr[2]
















