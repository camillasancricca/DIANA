from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math


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

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4),\
           corr[0],corr[1],corr[2]

def import_data_features(name, df, ML):

    with open(name+"_features.csv", "w") as file:
        file.write("name,tuples,features,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_dist,p_max_dist,p_min_dist,"+
                   "p_corr,max_pears,min_pears"+
                   "\n")

        features = str(import_features(df))
        features = features.replace("(", "")
        features = features.replace(")", "")
        features = features.replace(" ", "")
        file.write(name+","+features+"\n")

    file_path = name+"_features.csv"
    data_features = pd.read_csv(file_path)

    # with this method I calculate the features of the dataset for the imputation classifier
    create_classifier_features(name, df, ML)

    return data_features

def create_testdata(dataset, ML):

    test = import_data_features("dataset", dataset, ML)
    test = test.drop(columns="name")
    return test

def predict_ranking(KB,dataset_test,ML):

    #DecisionTree
    #KNN
    #NaiveBayes

    test = create_testdata(dataset_test, ML)
    class_name = "RANKING"

    train = KB[(KB["ML"] == ML)].drop(columns=["ML","name"])

    feature_cols = list(train.columns)



    feature_cols.remove(class_name)

    X = train[1:][feature_cols] # Features
    y = train[1:][class_name] # Target variable

    feature_columns = list(X.columns)

    X = StandardScaler().fit_transform(X)

    X = np.nan_to_num(X)
    X = pd.DataFrame(X, columns=feature_columns)

    test = np.nan_to_num(test)
    test = StandardScaler().fit_transform(test)
    test = pd.DataFrame(test, columns=feature_columns)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf = clf.fit(X, y)
    pred = clf.predict(test)
    #pred_proba = clf.predict_proba(test)

    print(pred)
    #print(pred_proba)
    return pred


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


def import_classifier_features(df):

    num = len(list(df.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool', 'object']).columns))

    rows = df.shape[0]
    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(df.duplicated().sum() / rows,
                                                                         4), df.memory_usage().sum(), \
        round(df.nunique().mean() / rows, 4), round(df.nunique().max() / rows, 4), round(df.nunique().min() / rows, 4), \
        round(sum(den) / len(den), 4), round(max(den), 4), round(min(den), 4), \
        round(sum(en) / len(en), 4), round(max(en), 4), round(min(en), 4), \
        corr[0], corr[1], corr[2]


def create_classifier_features(name, df, ML_model):

    file = open("dataset_classifier_features.csv", "w")

    file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
               "p_avg_distinct,p_max_distinct,p_min_distinct," +
               "avg_density,max_density,min_density," +
               "avg_entropy,max_entropy,min_entropy," +
               "p_correlated_features,max_pearson,min_pearson," +
               "ML_ALGORITHM" +
               "\n")

    features = str(import_classifier_features(df))
    features = features.replace("(", "")
    features = features.replace(")", "")
    features = features.replace(" ", "")

    file.write(name + "," + features + "," + ML_model.lower() + "\n")

    file.close()




# if __name__ == '__main__':

    # data = pd.read_csv("apps/datasets/iris.csv")
    # kb = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    # string = str(predict_ranking(kb, data, "DT"))
    # string = string.replace("[", "")
    # string = string.replace("]", "")
    # string = string.replace("'", "")
    # string = string.split()
    # print(string)

