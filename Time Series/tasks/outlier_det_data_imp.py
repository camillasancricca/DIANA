#TEST OUTLIER DETECTION AND CORRECTION

import pandas as pd
import pickle
from kafka import KafkaConsumer, TopicPartition
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pdb
from Lib import preproc_lib as pp
from Lib.outlier_det.MAD import MAD
from Lib.outlier_det.LOF import LOF
from Lib.outlier_det.z_score import z_score
from Lib.outlier_det.IForest import isoforest
from Lib.outlier_det.hst import HST
from Lib.imputation.LOCF import LOCF
from Lib.imputation.interpolation import Interpolation
from Lib import dq_lib as dq
from Lib import profiling_lib as lb
from Lib import kll
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import time
from Lib.khh import KHeavyHitters  # it uses CMS to keep track
import warnings

def feature_importance(forest,columns,X_test,y_test,list_forest_importance):
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=columns)
    list_forest_importance.append(forest_importances)


def plot_feature_importance(list_forest_importance):
    df = pd.concat(list_forest_importance, axis=1)
    result_series = df.mean(axis=1)
    std_dev_series = df.std(axis=1)
    fig, ax = plt.subplots()
    result_series.plot.bar(yerr=std_dev_series, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.ylim(top=0.9,bottom=-0.1)
    for i, v in enumerate(result_series):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    plt.savefig("../Img/1.jpg")

#Compute r2 and f1 metrics
def evaluate(ml_method,od_method, imp_method, actual_values, predicted_values, percentage):
    if ml_method == 'regression':
        actual_values = [float(item) for item in actual_values]
        predicted_values = [float(item) for item in predicted_values]
        r2 = metrics.r2_score(actual_values, predicted_values)
        print("R2-score ", r2)
        save_results(od_method, imp_method, percentage, r2)
    elif ml_method == 'classification' or ml_method == 'uni_class':
        f1 = metrics.f1_score(actual_values, predicted_values, average='micro')
        print("F1-score ", f1)
        save_results(od_method,imp_method,percentage,f1)


#Save results in a csv file
def save_results(od_method, imp_method, percentage, r2):
    path = "../Risultati/Air_Quality/knn_regression_1.csv"
    cols = ["percentage", "outlier", "imputation", "r2"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[percentage, od_method, imp_method, r2]], columns=cols)
    df = pd.concat([df, row], ignore_index=True)


    df.to_csv(path, index=False)
    return None


def regression(df, slide, cols, actual_values, predicted_values, target_cols,list_forest_importance):
    df_c = df.copy()
    r_cols = cols.copy()
    r_cols.remove('date_time')
    r_cols = [x for x in r_cols if x not in target_cols]
    r_cols.remove('arrive_time')
    # df_c.dropna(axis=0, inplace=True)
    y = df_c.pop('PM2.5')
    df_c = df_c.fillna(0)
    # X = df_c[r_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_c[r_cols])
    standardized_df = pd.DataFrame(standardized_data, columns=r_cols)
    X = standardized_df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
    neigh = KNeighborsRegressor(n_neighbors=10)
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    #if (count-1) % 4320 == 0 or count == 1009:
    #    feature_importance(rf_reg,r_cols,X_test,y_test,list_forest_importance)
    if len(predicted_values) == 0:
        actual_values.extend(y_test.values)
        predicted_values.extend(pred.tolist())
    else:
        actual_values.extend(y_test.values[-slide:])
        predicted_values.extend(pred.tolist()[-slide:])

    return actual_values, predicted_values

def classification(df, slide, cols, actual_values, predicted_values, target_cols):
    if len(df)>3:
        df_c = df.copy()
        r_cols = cols.copy()
        if 'date_time' in r_cols:
            r_cols.remove('date_time')
        r_cols = [x for x in r_cols if x not in target_cols]
        r_cols.remove('arrive_time')
        if 'date_time' in df.columns:
            df_c.drop(["date_time"], axis=1, inplace=True)
        df_c = df_c.fillna(0)
        y = df_c.pop('rain')
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df_c[r_cols])
        standardized_df = pd.DataFrame(standardized_data, columns=r_cols)
        X = standardized_df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
        rf_clf = RandomForestClassifier(random_state=42)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        if len(predicted_values) == 0:
            actual_values.extend(y_test.values)
            predicted_values.extend(pred.tolist())
        else:
            actual_values.extend(y_test.values[-slide:])
            predicted_values.extend(pred.tolist()[-slide:])

    return actual_values, predicted_values

def chlorine_classification(df, slide, cols, actual_values, predicted_values, target_cols):
    if len(df) > 0:
        df_c = df.copy()
        r_cols = cols.copy()
        if 'date_time' in r_cols:
            r_cols.remove('date_time')
        r_cols = [x for x in r_cols if x not in target_cols]
        r_cols.remove('arrive_time')
        if 'date_time' in df.columns:
            df_c.drop(["date_time"], axis=1, inplace=True)
        df_c = df_c.fillna(0)
        y = df_c.pop(0)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df_c[r_cols])
        standardized_df = pd.DataFrame(standardized_data, columns=r_cols)
        X = standardized_df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        if len(predicted_values) == 0:
            actual_values.extend(y_test.values)
            predicted_values.extend(pred.tolist())
        else:
            actual_values.extend(y_test.values[-slide:])
            predicted_values.extend(pred.tolist()[-slide:])

    return actual_values, predicted_values


warnings.filterwarnings("ignore")

scaler = StandardScaler()
start_time = time.time()
# collections.Iterable = collections.abc.Iterable
# collections.Mapping = collections.abc.Mapping
# collections.MutableSet = collections.abc.MutableSet
# collections.MutableMapping = collections.abc.MutableMapping

#set up variables
topic = 'csv-topic-31'
null_value = [' NA', '  NA', np.NaN, 'nan', ' ', '', None, 'NA']
date_time_format = '%Y-%m-%d %H:%M:%S'
ml_method = 'regression'

if ml_method == 'uni_class':
    columns = [i for i in range(167)]
    columns.append('arrive_time')
    types = ['float' for i in range(167)]
    types.append('string')
    target_cols=[0]
    not_target_cols = [i for i in range(167) if i not in [0,167]]
if ml_method == 'classification':
    columns = ['temp', 'dew_pnt', 'sea_lvl_press', 'visibility', 'avg_wind_spd', 'max_sustained_wind_spd', 'max_temp',
           'min_temp', 'rain','arrive_time']
    types = ['float','float','float','float','float','float','float','float','string','string']
    target_cols = ['rain']
    not_target_cols = [i for i in range(9) if i not in [len(columns)-1, len(columns)-2]]
if ml_method == 'regression':
    columns = ['date_time', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
              'RAIN', 'wd', 'WSPM', 'station', 'arrive_time']
    types = ['string', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
            'float', 'float', 'float', 'float', 'string', 'string']
    target_cols=['PM2.5']
    not_target_cols = [i for i in range(15) if i not in [0, 1, len(columns)-1, len(columns)-2]]

#methods that will be executed
outlier_detection_methods = ['none']
imputation_methods = ['none']
#outlier_detection_methods = ['z','lof','iforest','hst']
#imputation_methods = ['drop','LOCF','mean','interpolation']

#window size and sliding factor
ws = 1008
slide = 144

#quality level
percentage = 10

for od_method in outlier_detection_methods:
    for imp_method in imputation_methods:
        z = z_score(types)
        lof = LOF(n=5)
        iforest = isoforest()
        hst = HST()
        locf = LOCF()
        interp = Interpolation()

        # mad = []
        # quantile = []
        # for i in range(len(columns)):
        #    quantile.append(kll.KLL(256))
        #    mad.append(MAD())
        # od_method = "hst"
        # imp_method = "mean"

        count = 0
        c_outlier = 0
        outliers = []
        actual_values = []
        predicted_values = []
        list_forest_importance = []
        full_window_flag = False

        max_out = 0

        with open('../Datasets/outliers_index.pkl', 'rb') as pick:
            true_outliers = pickle.load(pick)

        consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
        tp = TopicPartition(topic, 0)
        consumer.assign([tp])
        consumer.seek_to_beginning()
        if od_method == 'arima':
            columns.append('count')
            df = pd.DataFrame(columns=columns)
        else:
            df = pd.DataFrame(columns=columns)
        p = consumer.position(tp)
        print("Connessione fatta")
        for message in consumer:
            row = str(message.value.decode('utf-8'))
            if row == 'finito':
                break

            outlier = False
            row = pp.row_str_preprocess(row)
            row = pp.row_type_preprocess(row, columns, types, null_value)
            # row = pp.row_add_datetime(row, date_time_format)

            if od_method == 'none':
                df = pd.concat([df, pd.DataFrame([row],columns=columns)], ignore_index=True)
                if len(df) > ws:
                    full_window_flag = True

            if od_method == 'z':
                outlier = z.add_sample(row)
                if outlier == True:
                    outliers.append(count)
                    row_n = [row[0], row[1]]
                    row_n.extend([None] * (len(columns) - 4))
                    row_n.extend([row[len(columns)-2], row[len(columns)-1]])
                    df = pd.concat([df, pd.DataFrame([row_n],columns=columns)], ignore_index=True)
                else:
                    df = pd.concat([df, pd.DataFrame([row],columns=columns)], ignore_index=True)
                if len(df) > ws:
                    full_window_flag = True
            # lb.c_quant(quantile, row)
            # _, q_50, _ = lb.comp_quantiles(quantile)
            # for col in range(len(columns)):
            #    if row[col] is not None and (types[col] == 'int' or types[col] == 'float'):
            #        o = mad[col].add_sample(q_50[col], row[col])
            #        if o is True:
            #            outlier = True
            if od_method == 'lof':
                window = ws
                # row = np.array(row[1:-1]).reshape(1, -1)
                # scaler = scaler.partial_fit(row)
                # row = scaler.transform(row)
                # df = df.append(pd.DataFrame(row, columns=columns[1:-1]))
                df = pd.concat([df, pd.DataFrame([row],columns=columns)], ignore_index=True)
                if len(df) > window:
                    full_window_flag = True
                    f_outliers = lof.compute(df.iloc[:, 1:-1], count, window, flag=True)
                    for idx in f_outliers:
                        df.iloc[idx, not_target_cols] = None

            if od_method == 'iforest':
                window = ws
                df = pd.concat([df, pd.DataFrame([row],columns=columns)], ignore_index=True)
                if count == 144:
                    iforest.fit(df)
                if len(df) > window:
                    full_window_flag = True
                    f_outliers = iforest.predict(df, count, window, flag=True)
                    for idx in f_outliers:
                        df.iloc[idx, not_target_cols] = None

            if od_method == 'hst':
                window = ws
                df_s = pd.DataFrame([row], columns=columns)
                o = hst.add_sample(df_s)
                if o:
                    outliers.append(count)
                    row_n = [row[0], row[1]]
                    row_n.extend([None] * (len(columns) - 4))
                    row_n.extend([row[len(columns)-2], row[len(columns)-1]])
                    df = pd.concat([df, pd.DataFrame([row_n],columns=columns)], ignore_index=True)
                else:
                    df = pd.concat([df, pd.DataFrame([row],columns=columns)], ignore_index=True)
                if len(df) > ws:
                    full_window_flag = True

            count += 1

            if imp_method == 'drop':
                df = df.dropna()
            if imp_method == 'LOCF':
                if full_window_flag:
                    df = df.ffill()
                    df = df.bfill()
            if imp_method == 'mean':
                # df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
                if full_window_flag:
                    column_means = df.iloc[:,1:-2].mean()
                    df.iloc[:,1:-2] = df.iloc[:,1:-2].fillna(column_means)
            if imp_method == 'interpolation':
                # df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
                if full_window_flag:
                    df = interp.interpolate(df, ml_method)

            if full_window_flag:
                if ml_method == 'regression':
                    actual_values, predicted_values = regression(df, slide, columns, actual_values, predicted_values, target_cols,list_forest_importance)
                elif ml_method == 'classification':
                    actual_values, predicted_values = classification(df, slide, columns, actual_values, predicted_values,
                                                             target_cols)
                elif ml_method == 'uni_class':
                    actual_values, predicted_values = chlorine_classification(df, slide, columns, actual_values,
                                                                     predicted_values,
                                                                     target_cols)

                df = df.tail(-slide)
                full_window_flag = False

            if count % 1000 == 0:
                print("C: ", count)

        evaluate(ml_method, od_method, imp_method, actual_values, predicted_values, percentage)
        #plot_feature_importance(list_forest_importance)