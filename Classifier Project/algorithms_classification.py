import pandas as pd
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder




def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)


def classification(dataset, class_name, classifier):

    # sostituito dal mio preprocessing, in basso
    """
    feature_cols = list(dataset.columns)
    feature_cols.remove(class_name)

    X = dataset[1:][feature_cols] # Features
    y = dataset[1:][class_name] # Target variable

    numeric_columns=list(X.select_dtypes(include=['int64','float64']).columns)
    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode_and_bind(X,col)

    feature_columns = list(X.columns)

    if len(numeric_columns)!=0 and len(categorical_columns)==0:
        X = StandardScaler().fit_transform(X)

    X = np.nan_to_num(X)
    X = pd.DataFrame(X, columns=feature_columns)

    """

    # X, y = dataset_preprocessing(dataset, class_name)
    X, y = dataset_preprocessing2(dataset, class_name)


    clf = DecisionTreeClassifier()

    #Choose classifier
    if classifier == "dt":
        clf = DecisionTreeClassifier()
    elif classifier == "knn":
        clf = KNeighborsClassifier()
    elif classifier == "nb":
        clf = GaussianNB()
    elif classifier == "sgd":
        clf = SGDClassifier()
    elif classifier == "svm":
        clf = SVC(kernel="linear")
    elif classifier == "svm-rbf":
        clf = SVC()
    elif classifier == "gpc":
        clf = GaussianProcessClassifier()
    elif classifier == "rf":
        clf = RandomForestClassifier()
    elif classifier == "ada":
        clf = AdaBoostClassifier()
    elif classifier == "bag":
        clf = BaggingClassifier()
    elif classifier == "lr":
        return logisticRegression(dataset, class_name)

    dt_fit = clf.fit(X, y)

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    dt_scores = cross_val_score(dt_fit, X, y, cv = cv, scoring="f1_weighted")
    # print(dt_scores.mean())
    return dt_scores.mean()


def logisticRegression(dataset, class_name):

    # sostituito dal mio preprocessing, in basso
    """
    feature_cols = list(dataset.columns)
    feature_cols.remove(class_name)

    X = dataset[1:][feature_cols] # Features
    y = dataset[1:][class_name] # Target variable

    numeric_columns=list(X.select_dtypes(include=['int64','float64']).columns)
    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode_and_bind(X,col)

    feature_columns = list(X.columns)

    if len(numeric_columns)!=0:
        X = X.where(X.notnull(), 0)
        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X, columns = feature_columns)

    """

    # X, y = dataset_preprocessing(dataset, class_name)
    X, y = dataset_preprocessing2(dataset, class_name)

    #Choose classifier
    clf = LogisticRegression(max_iter=1000)

    dt_fit = clf.fit(X, y)

    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    dt_scores = cross_val_score(dt_fit, X, y, cv = cv, scoring="f1_weighted")
    # print(dt_scores.mean())
    return dt_scores.mean()


# Aggiunti da Enrico

# Traina un modello. Funziona solo per dataset numerici (forse funziona con tutti).
def train_model(X, y, classifier):

    # X, y = dataset_preprocessing(dataset, class_name)

    clf = DecisionTreeClassifier()

    #Choose classifier
    if classifier == "dt":
        clf = DecisionTreeClassifier()
    elif classifier == "knn":
        clf = KNeighborsClassifier()
    elif classifier == "nb":
        clf = GaussianNB()
    elif classifier == "sgd":
        clf = SGDClassifier()
    elif classifier == "svm":
        clf = SVC(kernel="linear")
    elif classifier == "svm-rbf":
        clf = SVC()
    elif classifier == "gpc":
        clf = GaussianProcessClassifier()
    elif classifier == "rf":
        clf = RandomForestClassifier()
    elif classifier == "ada":
        clf = AdaBoostClassifier()
    elif classifier == "bag":
        clf = BaggingClassifier()
    elif classifier == "lr":
        clf = LogisticRegression(max_iter=1000)
        # return train_logisticRegression(dataset, class_name)

    dt_fit = clf.fit(X, y)

    # cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    # dt_scores = cross_val_score(dt_fit, X, y, cv = cv, scoring="f1_weighted")
    # print(dt_scores.mean())
    # return dt_scores.mean()

    # I train the model and then I return it
    return dt_fit


# in questo dataset preprocessing si fa scaling delle numerical features e encoding delle categoriche
def dataset_preprocessing(dataset, class_name):
    feature_cols = list(dataset.columns)
    feature_cols.remove(class_name)

    X = dataset[0:][feature_cols]  # Features
    y = dataset[0:][class_name]  # Target variable

    numeric_columns = list(X.select_dtypes(include=['int64', 'float64']).columns)
    categorical_columns = list(X.select_dtypes(include=['bool', 'object']).columns)

    # in caso di dataset misti uso ordinal encoder, altrimenti uso sempre il solito encode and bind
    if len(numeric_columns) != 0 and len(categorical_columns) != 0:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(X[categorical_columns])
        X[categorical_columns] = oe.transform(X[categorical_columns])
    else:
        for col in X.columns:
            if col in categorical_columns:
                X = encode_and_bind(X, col)

    feature_columns = list(X.columns)

    if len(numeric_columns) != 0:
        numeric_dataset = X[0:][numeric_columns]
        numeric_dataset = StandardScaler().fit_transform(numeric_dataset)
        numeric_dataset = pd.DataFrame(numeric_dataset, columns=numeric_columns)
        X[numeric_columns] = numeric_dataset[numeric_columns]

    X = np.nan_to_num(X)
    X = pd.DataFrame(X, columns=feature_columns)

    return X, y


# prova
def dataset_preprocessing2(dataset, class_name):
    feature_cols = list(dataset.columns)
    feature_cols.remove(class_name)

    X = dataset[0:][feature_cols]  # Features
    y = dataset[0:][class_name]  # Target variable

    numeric_columns = list(X.select_dtypes(include=['int64', 'float64']).columns)
    categorical_columns = list(X.select_dtypes(include=['bool', 'object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode_and_bind(X, col)

    feature_columns = list(X.columns)

    if len(numeric_columns) != 0:
        numeric_dataset = X[0:][numeric_columns]
        numeric_dataset = StandardScaler().fit_transform(numeric_dataset)
        numeric_dataset = pd.DataFrame(numeric_dataset, columns=numeric_columns)
        X[numeric_columns] = numeric_dataset[numeric_columns]

    # X = np.nan_to_num(X)
    X = X.fillna(0)
    X = pd.DataFrame(X, columns=feature_columns)

    return X, y


# forse è da togliere
def train_logisticRegression(dataset, class_name):

    feature_cols = list(dataset.columns)
    feature_cols.remove(class_name)

    X = dataset[1:][feature_cols] # Features
    y = dataset[1:][class_name] # Target variable

    numeric_columns=list(X.select_dtypes(include=['int64','float64']).columns)
    categorical_columns=list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode_and_bind(X,col)

    feature_columns = list(X.columns)

    if len(numeric_columns)!=0:
        X = X.where(X.notnull(), 0)
        X = StandardScaler().fit_transform(X)
        X = pd.DataFrame(X, columns = feature_columns)

    #Choose classifier
    clf = LogisticRegression(max_iter=1000)

    dt_fit = clf.fit(X, y)

    # cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
    # dt_scores = cross_val_score(dt_fit, X, y, cv = cv, scoring="f1_weighted")
    # print(dt_scores.mean())
    # return dt_scores.mean()

    return dt_fit















