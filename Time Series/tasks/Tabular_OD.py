import pickle

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


def save_results(ml_method, od_method, imp_method, percentage, r2):
    path = "../Risultati/Electrical/rf_classification_1.csv"
    cols = ["percentage", "ML", "outlier", "imputation", "r2"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[percentage, ml_method, od_method, imp_method, r2]], columns=cols)
    df = df.append(row)

    df.to_csv(path, index=False)
    return None


lof = LocalOutlierFactor(n_neighbors=5)
iforest = IsolationForest(random_state=1)
outlier_detection_methods = ['z','lof','iforest']
ml_methods = ['clf']
target_cols = ['stab', 'stabf']
percentage = 0

with open('../Datasets/outliers_index_1.pkl', 'rb') as pick:
    true_outliers = pickle.load(pick)

for od_method in outlier_detection_methods:
    df = pd.read_csv('../Datasets/Electrical_injected_outliers_1.csv')
    scaler = StandardScaler()

    if od_method == 'z':
        z_scores = df.iloc[:, :-2].apply(zscore)
        threshold = 3
        outliers = z_scores.abs() > threshold
        outliers = outliers.any(axis=1)
        outliers = outliers.index[outliers].tolist()
        df.iloc[outliers, :-2] = None

    if od_method == 'lof':
        data = df.iloc[:, :-2].to_numpy()
        data = scaler.fit_transform(data)
        outliers = lof.fit_predict(data)
        outliers = np.where(outliers == -1)[0]
        df.iloc[outliers, :-2] = None

    if od_method == 'iforest':
        data = df.iloc[:, :-2].to_numpy()
        outliers = lof.fit_predict(data)
        outliers = np.where(outliers == -1)[0]
        df.iloc[outliers, :-2] = None

    common_values = np.intersect1d(true_outliers, outliers)
    if len(common_values) > 0:
        # Compute precision
        precision = len(common_values) / len(outliers)

        # Compute recall
        recall = len(common_values) / len(true_outliers)

        f1 = (2 * precision * recall) / (precision + recall)

    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    print ("Detected: ", len(outliers))
    print("Common values: ", len(common_values))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score ", f1)
#    save_results(od_method, percentage, precision, recall, f1)
