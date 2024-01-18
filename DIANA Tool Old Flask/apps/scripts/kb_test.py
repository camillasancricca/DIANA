from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

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

def import_data_features(name, df):

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

    return data_features

def create_testdata(dataset):

    test = import_data_features("dataset", dataset)
    test = test.drop(columns="name")
    return test

def predict_ranking(KB,dataset_test,ML):

    test = create_testdata(dataset_test)
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

    clf = KNeighborsClassifier()
    clf = clf.fit(X, y)
    pred = clf.predict(test)
    #pred_proba = clf.predict_proba(test)

    print(pred)
    #print(pred_proba)
    return pred

# if __name__ == '__main__':

    # data = pd.read_csv("apps/datasets/iris.csv")
    # kb = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    # string = str(predict_ranking(kb, data, "DT"))
    # string = string.replace("[", "")
    # string = string.replace("]", "")
    # string = string.replace("'", "")
    # string = string.split()
    # print(string)

