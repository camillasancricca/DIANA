import random

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from joblib import dump, load


def main_classifier_whole_dataset():

    dataset = pd.read_csv("KB_whole_datasets_v2.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    training_set = dataset[dataset["name"].isin(["iris", "wine", "letter", "car", "users", "bank", "adult"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    feature_cols = list(dataset.columns)
    feature_cols.remove("name")
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_bfill', 'impute_ffill', 'impute_knn', 'impute_linear_and_logistic', 'impute_mean',
               'impute_median', 'impute_mice', 'impute_mode', 'impute_random', 'impute_soft', 'impute_standard',
               'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


def main_classifier_sancricca():

    dataset = pd.read_csv("KB_sancricca.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM', 'ERROR_TYPE'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())



    training_set = dataset[dataset["name"].isin(["iris", "wine", "letter", "car", "users", "bank", "adult"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    feature_cols = list(dataset.columns)
    feature_cols.remove("name")
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_bfill', 'impute_ffill', 'impute_knn', 'impute_linear_and_logistic', 'impute_mean',
               'impute_median', 'impute_mice', 'impute_mode', 'impute_random', 'impute_soft', 'impute_standard',
               'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
    cm_display.plot()
    plt.show()


def main_classifier_single_column():

    dataset = pd.read_csv("KB_single_columns_v2.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    training_set = dataset[dataset["name"].isin(["iris", "wine", "car", "users", "bank"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)

    # ora sostituisco i valori nan con 0
    # devo sostituire anche i valori None con 0

    X_train = X_train.replace(to_replace=["None"], value=np.nan)
    X_test = X_test.replace(to_replace=["None"], value=np.nan)

    # X_train = pd.DataFrame(X_train, columns=feature_cols)
    # X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_bfill', 'impute_ffill', 'impute_knn', 'impute_linear_regression', 'impute_logistic_regression',
               'impute_mean', 'impute_median', 'impute_mode', 'impute_random', 'impute_standard', 'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


# without the rows with bfill e ffill
def main_classifier_single_column_no_ffill():

    dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset)

    training_set = dataset[dataset["name"].isin(["iris", "wine", "car", "users", "bank", "letter", "adult"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    # training_set = dataset[dataset["name"].isin(["iris", "cancer", "car", "users", "bank", "letter", "german"])]
    # test_set = dataset[dataset["name"].isin(["wine", "adult"])]

    # tolgo righe con ffill e bfill
    # training_set = training_set[training_set.BEST_METHOD != "impute_ffill"]
    # training_set = training_set[training_set.BEST_METHOD != "impute_bfill"]
    # test_set = test_set[test_set.BEST_METHOD != "impute_ffill"]
    # test_set = test_set[test_set.BEST_METHOD != "impute_bfill"]

    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)

    # ora sostituisco i valori nan con 0
    # devo sostituire anche i valori None con 0

    X_train = X_train.replace(to_replace=["None"], value=np.nan)
    X_test = X_test.replace(to_replace=["None"], value=np.nan)

    # X_train = pd.DataFrame(X_train, columns=feature_cols)
    # X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_regression', 'impute_logistic_regression',
               'impute_mean', 'impute_median', 'impute_mode', 'impute_random', 'impute_standard', 'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


# without the rows with bfill e ffill
def main_classifier_whole_dataset_no_ffill():
    # dataset = pd.read_csv("KB_whole_datasets_v2.csv")
    dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    # training_set = dataset[dataset["name"].isin(["iris", "wine", "letter", "car", "users", "bank", "adult", "soybean", "mushrooms"])]
    # test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    training_set = dataset[dataset["name"].isin(["iris", "cancer", "letter", "car", "users", "german", "adult", "soybean", "mushrooms"])]
    test_set = dataset[dataset["name"].isin(["wine", "bank"])]

    # tolgo righe con ffill e bfill
    # training_set = training_set[training_set.BEST_METHOD != "impute_ffill"]
    # training_set = training_set[training_set.BEST_METHOD != "impute_bfill"]
    # test_set = test_set[test_set.BEST_METHOD != "impute_ffill"]
    # test_set = test_set[test_set.BEST_METHOD != "impute_bfill"]

    feature_cols = list(dataset.columns)
    feature_cols.remove("name")
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_and_logistic', 'impute_mean',
               'impute_median', 'impute_mice', 'impute_mode', 'impute_random', 'impute_soft', 'impute_standard',
               'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


# questa funzione considera solo le rows con 50% missing
def main_classifier_whole_dataset_only_some_rows():
    # dataset = pd.read_csv("KB_whole_datasets_v2.csv")
    dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # prendo una riga ogni 5
    dataset = dataset.iloc[::5, :]

    # prendo solo i missing 50% e i 10%
    # dataset1 = dataset.iloc[::5, :]
    # dataset2 = dataset.iloc[4::5, :]

    # dataset = pd.concat([dataset1, dataset2], axis=0)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    # training_set = dataset[dataset["name"].isin(["iris", "wine", "letter", "car", "users", "bank", "adult"])]
    # test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    training_set = dataset[dataset["name"].isin(["iris", "cancer", "car", "users", "german", "adult"])]
    test_set = dataset[dataset["name"].isin(["wine", "bank", "letter"])]

    feature_cols = list(dataset.columns)
    feature_cols.remove("name")
    feature_cols.remove(class_name)
    # feature_cols.remove("%missing")

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_and_logistic', 'impute_mean',
               'impute_median', 'impute_mice', 'impute_mode', 'impute_random', 'impute_soft', 'impute_standard',
               'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


def main_classifier_single_column_only_some_rows():

    dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # prendo una riga ogni 5
    dataset = dataset.iloc[::5, :]

    # prendo solo i missing 50% e i 10%
    # dataset1 = dataset.iloc[::5, :]
    # dataset2 = dataset.iloc[4::5, :]

    # dataset = pd.concat([dataset1, dataset2], axis=0)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset)

    training_set = dataset[dataset["name"].isin(["iris", "wine", "car", "users", "bank", "letter", "adult", "soybean", "mushrooms"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    # training_set = dataset[dataset["name"].isin(["iris", "cancer", "car", "users", "bank", "letter", "german"])]
    # test_set = dataset[dataset["name"].isin(["wine", "adult"])]

    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)

    # ora sostituisco i valori nan con 0
    # devo sostituire anche i valori None con 0

    X_train = X_train.replace(to_replace=["None"], value=np.nan)
    X_test = X_test.replace(to_replace=["None"], value=np.nan)

    # X_train = pd.DataFrame(X_train, columns=feature_cols)
    # X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_regression', 'impute_logistic_regression',
               'impute_mean', 'impute_median', 'impute_mode', 'impute_random', 'impute_standard', 'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


def main_classifier_single_column_feature_importance():

    dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset)

    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')
    feature_cols.remove(class_name)

    X = dataset[0:][feature_cols]  # Features
    y = dataset[0:][class_name]  # Target variable

    print(X.isnull().sum().sum())

    X = X.replace(to_replace=["None"], value=np.nan)

    print(X.isnull().sum().sum())

    X = X.fillna(0)

    print(X.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=feature_cols)

    print(X)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    forest = RandomForestClassifier()

    forest.fit(X, y)

    importances = forest.feature_importances_

    # print(importances)

    indices = np.argsort(importances)[::-1]  # questo ti ritorna già ordine decrescente

    feature_columns = X.columns
    selected_columns = feature_columns[indices]
    importances = importances[indices]

    print(indices)
    print(importances)
    print(selected_columns)


def main_classifier_whole_dataset_feature_importance():

    dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset)

    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(class_name)

    X = dataset[0:][feature_cols]  # Features
    y = dataset[0:][class_name]  # Target variable

    print(X.isnull().sum().sum())

    X = X.replace(to_replace=["None"], value=np.nan)

    print(X.isnull().sum().sum())

    X = X.fillna(0)

    print(X.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=feature_cols)

    print(X)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    forest = RandomForestClassifier()

    forest.fit(X, y)

    importances = forest.feature_importances_

    # print(importances)

    indices = np.argsort(importances)[::-1]  # questo ti ritorna già ordine decrescente

    feature_columns = X.columns
    selected_columns = feature_columns[indices]
    importances = importances[indices]

    print(indices)
    print(importances)
    print(selected_columns)


def main_classifier_single_column_feature_selection():

    dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset)

    training_set = dataset[dataset["name"].isin(["iris", "wine", "car", "users", "bank", "letter", "adult"])]
    test_set = dataset[dataset["name"].isin(["cancer", "german"])]



    feature_cols = list(dataset.columns)
    feature_cols.remove('name')
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')
    feature_cols.remove(class_name)

    # feature selection manuale, tengo solo le features più importanti, che ho trovato con
    # random forest feature importance

    feature_cols.remove('min_pearson')
    feature_cols.remove(' n_tuples')
    feature_cols.remove('p_duplicates')
    feature_cols.remove('total_size')
    feature_cols.remove('max_pearson')
    feature_cols.remove('p_cat_var')
    feature_cols.remove('p_num_var')
    feature_cols.remove('p_correlated_features')

    # feature_cols.remove('%missing')

    """ # features da togliere
    'min_pearson', ' n_tuples',
    'p_duplicates', 'total_size', 'max_pearson', 'p_cat_var', 'p_num_var',
    'column_type_categorical', 'column_type_numerical',
    'p_correlated_features'
    """

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # X_train = np.nan_to_num(X_train)
    # X_test = np.nan_to_num(X_test)

    # ora sostituisco i valori nan con 0
    # devo sostituire anche i valori None con 0

    X_train = X_train.replace(to_replace=["None"], value=np.nan)
    X_test = X_test.replace(to_replace=["None"], value=np.nan)

    # X_train = pd.DataFrame(X_train, columns=feature_cols)
    # X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_regression', 'impute_logistic_regression',
               'impute_mean', 'impute_median', 'impute_mode', 'impute_random', 'impute_standard', 'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


def main_classifier_whole_dataset_mode():
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_mode.csv")
    dataset = pd.read_csv("KB_whole_datasets_total_mode.csv")

    # class_name = dataset.columns[-1]
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    # training_set = dataset[dataset["name"].isin(["iris", "wine", "letter", "car", "users", "bank", "adult", "soybean", "mushrooms"])]
    # test_set = dataset[dataset["name"].isin(["cancer", "german"])]

    training_set = dataset[dataset["name"].isin(["iris", "cancer", "wine", "car", "users", "german",
                                                 "soybean", "mushrooms", "breast-cancer", "drug", "ecoli", "gender",
                                                 "mobile", "stars"])]
    test_set = dataset[dataset["name"].isin(["letter", "bank", "adult"])]


    feature_cols = list(dataset.columns)
    feature_cols.remove("name")
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    X_test = test_set[0:][feature_cols]  # Features
    y_test = test_set[0:][class_name]  # Target variable

    print(X_train.shape)
    print(X_test.shape)

    print(X_train.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train.isnull().sum().sum())

    # ora faccio scaling del dataset

    scaler = MinMaxScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train, columns=feature_cols)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    print(X_train)

    # ok quindi ora ho il training set e il test set pronti

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression()

    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, y_pred))
    # print("Precision: ", precision_score(y_test, y_pred))
    # print("Recall: ", recall_score(y_test, y_pred))
    # print("f1 : ", f1_score(y_test, y_pred))

    classes = ['impute_knn', 'impute_linear_and_logistic', 'impute_mean',
               'impute_median', 'impute_mice', 'impute_mode', 'impute_random', 'impute_soft', 'impute_standard',
               'impute_std', 'no_impute']
    confusionMatrix = confusion_matrix(y_test, y_pred, labels=classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    cm_display.plot()
    plt.show()


# questa funzione è per whole dataset
def main_classifier_cross_validation():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_mode.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_mode.csv")
    #  dataset = pd.read_csv("KB_whole_datasets_no_ffill_preprocessing.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_preprocessing.csv")
    dataset = pd.read_csv("KB_whole_datasets_no_fill_preprocessing_mode.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)
    # feature_cols.remove("name")
    # feature_cols.remove(class_name)

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression(penalty='l2', C=1)
    # clf = LogisticRegression()



    # ---------- ora facciamo cross validation

    # lista datasets

    datasets = ["iris", "cancer", "wine", "car", "users", "german", "soybean", "mushrooms", "letter", "bank", "adult"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler = RobustScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_sancricca():

    # inserire qui codice per dataset preprocessing e classifier

    dataset = pd.read_csv("KB_sancricca.csv")
    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM', 'ERROR_TYPE'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)
    # feature_cols.remove("name")
    # feature_cols.remove(class_name)

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression()



    # ---------- ora facciamo cross validation

    # lista datasets
    datasets = ["iris", "cancer", "wine", "car", "users", "german", "soybean", "mushrooms", "letter", "bank", "adult"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_single_column():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # dataset = pd.read_csv("KB_single_columns_total.csv")
    dataset = pd.read_csv("KB_single_columns_no_fill_preprocessing_mode.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)

    # rimuovo colonne inutili
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0

    dataset = dataset.replace(to_replace=["None"], value=np.nan)

    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression(penalty = 'l2')
    clf = LogisticRegression()



    # ---------- ora facciamo cross validation

    # lista datasets

    datasets = ["iris", "cancer", "wine", "car", "users", "german", "letter", "bank", "adult", "soybean", "mushrooms"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_filtering():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_mode.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total.csv")
    dataset = pd.read_csv("KB_whole_datasets_total_mode.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # filtro in base al ML method, scelgo dt
    dataset = dataset[dataset["ML_ALGORITHM"] == "nb"]

    # one hot encoding delle categorical features
    # dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)
    # feature_cols.remove("name")
    # feature_cols.remove(class_name)

    # rimuovo colonna con ML algorithm tanto mo è sempre dt
    feature_cols.remove("ML_ALGORITHM")

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = DecisionTreeClassifier()
    clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression()



    # ---------- ora facciamo cross validation


    # lista datasets
    datasets = ["iris", "cancer", "wine", "car", "users", "german", "soybean", "mushrooms",
                "letter", "bank", "adult", "breast-cancer", "drug", "ecoli", "gender", "mobile", "stars"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_whole_three():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_mode.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_mode.csv")
    dataset = pd.read_csv("KB_whole_datasets_no_ffill_preprocessing.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_preprocessing.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_fill_preprocessing_mode.csv")

    # carico la kb 2
    KB2 = pd.read_csv("KB_whole_datasets_three.csv")
    # KB2 = pd.read_csv("KB_whole_datasets_three_aggregated.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)
    # feature_cols.remove("name")
    # feature_cols.remove(class_name)

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    clf = KNeighborsClassifier(n_neighbors=4)
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression(penalty='l2', C=1)
    # clf = LogisticRegression()



    # ---------- ora facciamo cross validation

    # lista datasets

    datasets = ["iris", "cancer", "wine", "users", "german", "soybean", "mushrooms", "letter", "bank"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]
        KB2_test_set = KB2[KB2["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler = RobustScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # accuracy = accuracy_score(y_test, y_pred)

        best_methods1 = KB2_test_set["BEST_METHOD1"]
        best_methods2 = KB2_test_set["BEST_METHOD2"]
        best_methods3 = KB2_test_set["BEST_METHOD3"]

        mask1 = np.where(best_methods1 == y_pred, True, False)
        mask2 = np.where(best_methods2 == y_pred, True, False)
        mask3 = np.where(best_methods3 == y_pred, True, False)

        mask_tot = [a or b or c for a, b, c in zip(mask1, mask2, mask3)]

        accuracy = sum(mask_tot) / len(mask_tot)

        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_single_column_three():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # dataset = pd.read_csv("KB_single_columns_total.csv")
    dataset = pd.read_csv("KB_single_columns_no_fill_preprocessing.csv")
    # dataset = pd.read_csv("KB_single_columns_no_fill_preprocessing_mode.csv")

    # carico la kb 2
    KB2 = pd.read_csv("KB_single_columns_three.csv")
    # KB2 = pd.read_csv("KB_single_columns_three_aggregated.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)

    # rimuovo colonne inutili
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0

    dataset = dataset.replace(to_replace=["None"], value=np.nan)

    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=20)
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression(penalty = 'l2')
    clf = LogisticRegression(max_iter=1000)



    # ---------- ora facciamo cross validation

    # lista datasets

    datasets = ["iris", "cancer", "wine", "car", "users", "german", "soybean", "mushrooms"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]
        KB2_test_set = KB2[KB2["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        y_train = training_set[0:][class_name]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()
        # scaler = RobustScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # accuracy = accuracy_score(y_test, y_pred)

        best_methods1 = KB2_test_set["BEST_METHOD1"]
        best_methods2 = KB2_test_set["BEST_METHOD2"]
        best_methods3 = KB2_test_set["BEST_METHOD3"]

        mask1 = np.where(best_methods1 == y_pred, True, False)
        mask2 = np.where(best_methods2 == y_pred, True, False)
        mask3 = np.where(best_methods3 == y_pred, True, False)

        mask_tot = [a or b or c for a, b, c in zip(mask1, mask2, mask3)]

        accuracy = sum(mask_tot) / len(mask_tot)

        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def main_classifier_cross_validation_single_column_three_2():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_single_columns_no_ffill.csv")
    # dataset = pd.read_csv("KB_single_columns_total.csv")
    # dataset = pd.read_csv("KB_single_columns_no_fill_preprocessing.csv")
    dataset = pd.read_csv("KB_single_columns_no_fill_preprocessing_mode.csv")

    # carico la kb 2
    # KB2 = pd.read_csv("KB_single_columns_three.csv")
    KB2 = pd.read_csv("KB_single_columns_three_aggregated.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['column_type', 'ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)

    # rimuovo colonne inutili
    feature_cols.remove(' column_name')
    feature_cols.remove('n_attributes')

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0

    dataset = dataset.replace(to_replace=["None"], value=np.nan)

    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    # clf = KNeighborsClassifier(n_neighbors=16)
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
    # clf = LogisticRegression(penalty = 'l2')
    # clf = LogisticRegression(max_iter=1000)



    # ---------- ora facciamo cross validation

    # lista datasets

    datasets = ["iris", "wine", "car", "users", "german", "letter", "soybean", "mushrooms"]

    accuracy_sum = 0.0  # somma delle accuracy per poi fare l'accuracy media

    for test_set_name in datasets:

        feature_cols = list(dataset.columns)

        training_set = dataset[dataset["name"] != test_set_name]
        KB2_training_set = KB2[KB2["name"] != test_set_name]
        test_set = dataset[dataset["name"] == test_set_name]
        KB2_test_set = KB2[KB2["name"] == test_set_name]

        feature_cols.remove("name")
        feature_cols.remove(class_name)

        X_train = training_set[0:][feature_cols]  # Features
        # y_train = training_set[0:][class_name]  # Target variable
        y_train = KB2_training_set[0:]["BEST_METHOD1"]  # Target variable

        X_test = test_set[0:][feature_cols]  # Features
        # y_test = test_set[0:][class_name]  # Target variable

        print(X_train.shape)
        print(X_test.shape)

        # faccio scaling

        scaler = MinMaxScaler()
        # scaler = RobustScaler()
        # scaler = StandardScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # fitto il modello

        model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # accuracy = accuracy_score(y_test, y_pred)

        best_methods1 = KB2_test_set["BEST_METHOD1"]
        best_methods2 = KB2_test_set["BEST_METHOD2"]
        best_methods3 = KB2_test_set["BEST_METHOD3"]

        mask1 = np.where(best_methods1 == y_pred, True, False)
        mask2 = np.where(best_methods2 == y_pred, True, False)
        mask3 = np.where(best_methods3 == y_pred, True, False)

        mask_tot = [a or b or c for a, b, c in zip(mask1, mask2, mask3)]

        accuracy = sum(mask_tot) / len(mask_tot)

        print("Accuracy: ", accuracy)
        # print("Precision: ", precision_score(y_test, y_pred))
        # print("Recall: ", recall_score(y_test, y_pred))
        # print("f1 : ", f1_score(y_test, y_pred))

        accuracy_sum = accuracy_sum + accuracy

    average_accuracy = accuracy_sum / len(datasets)
    print(average_accuracy)


def aggregate_KB2_whole():

    KB2 = pd.read_csv("KB_whole_datasets_three.csv")
    file_KB_aggregated = open("KB_whole_datasets_three_aggregated.csv", "w")

    for dataset_name in KB2["name"].unique().tolist():

        for ML_method in ["dt", "lr", "knn", "nb"]:

            df = KB2[KB2["name"] == dataset_name]
            df = df[df["ML_ALGORITHM"] == ML_method]

            best_methods1 = df["BEST_METHOD1"]
            best_methods2 = df["BEST_METHOD2"]
            best_methods3 = df["BEST_METHOD3"]

            scores = {}

            for method1 in best_methods1:
                if method1 in scores.keys():
                    scores[method1] = scores[method1] + 3
                else:
                    scores[method1] = 3
            for method2 in best_methods2:
                if method2 in scores.keys():
                    scores[method2] = scores[method2] + 2
                else:
                    scores[method2] = 2
            for method3 in best_methods3:
                if method3 in scores.keys():
                    scores[method3] = scores[method3] + 1
                else:
                    scores[method3] = 1

            best_method = max(scores, key=scores.get)
            del scores[best_method]
            second_best_method = max(scores, key=scores.get)
            del scores[second_best_method]
            third_best_method = max(scores, key=scores.get)

            file_KB_aggregated.write(dataset_name + "," + ML_method + "," + best_method + "," +
                                     second_best_method + "," + third_best_method + "\n")

    file_KB_aggregated.close()


def aggregate_KB2_single():

    # KB2 = pd.read_csv("KB_single_columns_three_prova.csv")
    # file_KB_aggregated = open("KB_single_columns_three_prova_aggregated.csv", "w")

    KB2 = pd.read_csv("KB_single_columns_three.csv")
    file_KB_aggregated = open("KB_single_columns_three_aggregated.csv", "w")

    file_KB_aggregated.write("name" + "," + " column_name" + "," + "ML_ALGORITHM" + "," + "BEST_METHOD1" + "," +
                             "BEST_METHOD2" + "," + "BEST_METHOD3" + "\n")

    for dataset_name in KB2["name"].unique().tolist():

        dataset_rows = KB2[KB2["name"] == dataset_name]

        for column_name in dataset_rows[" column_name"].unique().tolist():

            for ML_method in ["dt", "lr", "knn", "nb"]:
            # for ML_method in ["dt", "lr"]:

                # df = KB2[KB2["name"] == dataset_name]
                df = dataset_rows[dataset_rows[" column_name"] == column_name]
                df = df[df["ML_ALGORITHM"] == ML_method]

                best_methods1 = df["BEST_METHOD1"]
                best_methods2 = df["BEST_METHOD2"]
                best_methods3 = df["BEST_METHOD3"]

                scores = {}

                for method1 in best_methods1:
                    if method1 in scores.keys():
                        scores[method1] = scores[method1] + 3
                    else:
                        scores[method1] = 3
                for method2 in best_methods2:
                    if method2 in scores.keys():
                        scores[method2] = scores[method2] + 2
                    else:
                        scores[method2] = 2
                for method3 in best_methods3:
                    if method3 in scores.keys():
                        scores[method3] = scores[method3] + 1
                    else:
                        scores[method3] = 1

                best_method = max(scores, key=scores.get)
                del scores[best_method]
                second_best_method = max(scores, key=scores.get)
                del scores[second_best_method]
                third_best_method = max(scores, key=scores.get)

                file_KB_aggregated.write(dataset_name + "," + column_name + "," + ML_method + "," + best_method + "," +
                                         second_best_method + "," + third_best_method + "\n")

    file_KB_aggregated.close()



def main_classifier_whole_datasets_final_training():

    # inserire qui codice per dataset preprocessing e classifier

    # dataset = pd.read_csv("KB_whole_datasets_no_ffill.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_mode.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_mode.csv")
    # dataset = pd.read_csv("KB_whole_datasets_no_ffill_preprocessing.csv")
    # dataset = pd.read_csv("KB_whole_datasets_total_preprocessing.csv")
    dataset = pd.read_csv("KB_whole_datasets_no_fill_preprocessing_mode.csv")

    class_name = "BEST_METHOD"

    print(dataset)

    # one hot encoding delle categorical features
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # attento che qui ora non ho più ordine iniziale delle colonne, ma quelle encodate stanno alla fine

    print(dataset.head())

    feature_cols = list(dataset.columns)
    # feature_cols.remove("name")
    # feature_cols.remove(class_name)

    print(dataset.isnull().sum().sum())

    # ora sostituisco i valori nan con 0
    # dataset = np.nan_to_num(dataset)
    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset.isnull().sum().sum())

    # ora faccio scaling del dataset
    # scaler = MinMaxScaler()
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)

    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)

    print(dataset)

    clf = KNeighborsClassifier(n_neighbors=4)
    # clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier()
    # clf = LogisticRegression(penalty='l1', solver='liblinear')
    # clf = LogisticRegression(penalty='l2', C=1)
    # clf = LogisticRegression()



    # ---------- ora traino su tutti i datasets

    # lista datasets

    datasets = ["iris", "cancer", "wine", "users", "german", "soybean", "mushrooms", "letter", "bank"]

    feature_cols = list(dataset.columns)

    training_set = dataset

    feature_cols.remove("name")
    feature_cols.remove(class_name)

    X_train = training_set[0:][feature_cols]  # Features
    y_train = training_set[0:][class_name]  # Target variable

    print(X_train.shape)

    # faccio scaling

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    # scaler = RobustScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)

    X_train = pd.DataFrame(X_train, columns=feature_cols)

    # fitto il modello

    model = clf.fit(X_train, y_train)

    dump(model, 'trained_classifier.joblib')
    dump(scaler, 'trained_scaler.joblib')

    model = load('trained_classifier.joblib')

    y_pred = model.predict(X_train)
    print(y_pred)
    print(y_train)





if __name__ == '__main__':
    # main_classifier_whole_dataset()
    # main_classifier_sancricca()
    # main_classifier_single_column()
    # main_classifier_single_column_no_ffill()
    # main_classifier_whole_dataset_no_ffill()
    # main_classifier_whole_dataset_only_some_rows()
    # main_classifier_single_column_only_some_rows()
    # main_classifier_single_column_feature_importance()
    # main_classifier_whole_dataset_feature_importance()
    # main_classifier_single_column_feature_selection()
    # main_classifier_whole_dataset_mode()
    # main_classifier_cross_validation()
    # main_classifier_cross_validation_sancricca()
    # main_classifier_cross_validation_single_column()
    # main_classifier_cross_validation_filtering()
    main_classifier_cross_validation_whole_three()
    # main_classifier_cross_validation_single_column_three()
    # main_classifier_cross_validation_single_column_three_2()
    # aggregate_KB2_whole()
    # aggregate_KB2_single()
    # main_classifier_whole_datasets_final_training()





























