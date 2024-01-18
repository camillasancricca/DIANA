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
    cols = ["percentage","ML", "outlier", "imputation", "r2"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[percentage,ml_method, od_method, imp_method, r2]], columns=cols)
    df = df.append(row)

    df.to_csv(path, index=False)
    return None


lof = LocalOutlierFactor(n_neighbors=5)
iforest = IsolationForest(random_state=1)
#outlier_detection_methods = ['z']
data_imputation_methods = ['none','drop', 'locf', 'mean', 'interpolation']
outlier_detection_methods = ['none']
#data_imputation_methods = ['none']
ml_methods = ['clf']
target_cols = ['stab','stabf']
percentage = 0

for ml_method in ml_methods:
    for od_method in outlier_detection_methods:
        for imp_method in data_imputation_methods:
            df = pd.read_csv('../Datasets/Electrical_injected_missing_1.csv', na_values=' NA')
            scaler = StandardScaler()

            if od_method == 'z':
                z_scores = df.iloc[:,:-2].apply(zscore)
                threshold = 3
                outliers = z_scores.abs() > threshold
                outlier_rows = outliers.any(axis=1)
                outlier_rows = outlier_rows.index[outlier_rows].tolist()
                df.iloc[outlier_rows, :-2] = None

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

            if imp_method == 'drop':
                df = df.dropna()

            if imp_method == 'locf':
                df = df.ffill()

            if imp_method == 'mean':
                column_means = df.mean()
                df = df.fillna(column_means)

            if imp_method == 'interpolation':
                df = df.interpolate(method='linear')

            if ml_method == 'reg':
                df = df.fillna(0)
                df = df.drop(['stabf'], axis=1)
                y = df.pop('stab')
                X = df
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                                    shuffle=False)
                rf_reg = RandomForestRegressor(random_state=42)
                rf_reg.fit(X_train, y_train)
                pred = rf_reg.predict(X_test)
                r2 = metrics.r2_score(y_test, pred)
                print("R2-score ", r2)
                save_results(ml_method, od_method, imp_method, percentage, r2)

            if ml_method == 'clf':
                df = df.fillna(0)
                df = df.drop(['stab'], axis=1)
                y = df.pop('stabf')
                X = df
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                                    shuffle=False)
                rf_reg = RandomForestClassifier(random_state=42)
                rf_reg.fit(X_train, y_train)
                pred = rf_reg.predict(X_test)
                f1 = metrics.f1_score(y_test, pred, average='micro')
                print("F1-score ", f1)
#                save_results(ml_method, od_method, imp_method, percentage, f1)




