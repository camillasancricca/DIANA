#TEST OUTLIER TO STANDARD VALUES AFTER OUTLIER DETECTION

import pandas as pd
import pickle
from kafka import KafkaConsumer, TopicPartition
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from Lib import preproc_lib as pp
from Lib.outlier_det.MAD import MAD
from Lib.outlier_det.ARIMA import ARIMA
from Lib.outlier_det.LOF import LOF
from Lib.outlier_det.z_score import z_score
from Lib.outlier_det.IForest import isoforest
from Lib.outlier_det.deepant_support import deepant_support
from Lib.outlier_det.hst import HST
from Lib.imputation.LOCF import LOCF
from Lib.imputation.interpolation import Interpolation
from Lib import dq_lib as dq
from Lib import profiling_lib as lb
from Lib import kll
from Lib import eff_apriori

import time
from Lib.khh import KHeavyHitters  # it uses CMS to keep track
import warnings


def evaluate(od_method, actual_values, predicted_values, percentage):
    actual_values = [float(item) for item in actual_values]
    predicted_values = [float(item) for item in predicted_values]
    r2 = metrics.r2_score(actual_values, predicted_values)
    print("R2-score ", r2)
    save_results(od_method, percentage, r2)


def save_results(od_method, percentage, r2):
    path = "../Risultati/Air_Quality/outlier_detection_regression.csv"
    cols = ["percentage", "method", "r2"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[percentage, od_method, r2]], columns=cols)
    df = df.append(row)

    df.to_csv(path, index=False)
    return None


def regression(df, slide, cols, actual_values, predicted_values):
    df_c = df.copy()
    r_cols = cols.copy()
    r_cols.remove('date_time')
    r_cols.remove('PM2.5')
    r_cols.remove('arrive_time')
    if 'date_time' in df.columns:
        df_c.drop(["date_time"], axis=1, inplace=True)
    #df_c.dropna(axis=0, inplace=True)
    df_c = df_c.fillna(0)
    y = df_c.pop('PM2.5')
    #X = df_c[r_cols]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_c[r_cols])
    standardized_df = pd.DataFrame(standardized_data, columns=r_cols)
    X = standardized_df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    pred = rf_reg.predict(X_test)
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


topic = 'csv-topic'
# columns = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description', 'date_time',
#           'traffic_volume','arrive_date']
# types = ['string', 'float', 'float', 'float', 'int', 'string', 'string', 'string', 'int', 'string']
# columns = ["Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)",
#           "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "Temp", "RH", "AH", "arrive_time"]
# types = ["string", "string", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float",
#         "float", "float", "float", "string"]
columns = ['date_time', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
           'RAIN', 'wd', 'WSPM', 'station', 'arrive_time']
types = ['string', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
         'float', 'float', 'float', 'float', 'string', 'string']
null_value = [' NA', '  NA', np.NaN, 'nan', ' ', '', None]
date_time_format = '%Y-%m-%d %H:%M:%S'

outlier_detection_methods = ['iforest','hst']

ws = 336
slide = 48
percentage = 50

for od_method in outlier_detection_methods:
    z = z_score(types)
    lof = LOF(n=5)
    arima = ARIMA(columns=types)
    iforest = isoforest()
    deepant = deepant_support()
    hst = HST()

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
    not_target_cols = [i for i in range(15) if i not in [0, 1, 13, 14]]
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
            df = df.append(pd.DataFrame([row], columns=columns))
            if len(df) > ws:
                full_window_flag = True

        if od_method == 'z':
            outlier = z.add_sample(row)
            if outlier == True:
                outliers.append(count)
                row_n = [row[0], row[1]]
                row_n.extend([None] * (len(columns) - 4))
                row_n.extend([row[13], row[14]])
                df = df.append(pd.DataFrame([row_n], columns=columns))
            else:
                df = df.append(pd.DataFrame([row], columns=columns))
            if len(df) > ws:
                full_window_flag = True

        if od_method == 'lof':
            window = ws
            df = df.append(pd.DataFrame([row], columns=columns))
            if len(df) > window:
                full_window_flag = True
                f_outliers = lof.compute(df.iloc[:, 1:-1], count, window, flag=True)
                for idx in f_outliers:
                    df.iloc[idx, not_target_cols] = None

        if od_method == 'arima':
            window = ws
            if row[-2] == ' 0':
                row.append(count)
                r = pd.DataFrame([row], columns=columns, index=pd.DatetimeIndex([row[0]]))
                df = df.append(r)
            if count == 648:
                df = df.drop(['date_time', 'arrive_time'], axis=1)
                arima.fit(df)
            if len(df) > window:
                df = df.drop(['date_time', 'arrive_time'], axis=1)
                f_outliers = arima.compute_sample(df, count, window)
                outliers = np.union1d(outliers, f_outliers)

        if od_method == 'iforest':
            window = ws
            df = df.append(pd.DataFrame([row], columns=columns))
            if count == 144:
                iforest.fit(df)
            if len(df) > window:
                full_window_flag = True
                f_outliers = iforest.predict(df, count, window, flag=True)
                for idx in f_outliers:
                    df.iloc[idx, not_target_cols] = None

        if od_method == 'deepant':
            window = ws
            df = df.append(pd.DataFrame([row], columns=columns))
            if count == 999:
                deepant.fit(df.loc[:, 'TEMP'])
            if len(df) > window:
                f_outliers = deepant.predict(df.loc[:, 'TEMP'])
                outliers = np.union1d(outliers, f_outliers)

        if od_method == 'hst':
            window = ws
            df_s = pd.DataFrame([row], columns=columns)
            o = hst.add_sample(df_s)
            if o:
                outliers.append(count)
                row_n = [row[0], row[1]]
                row_n.extend([None] * (len(columns) - 4))
                row_n.extend([row[13], row[14]])
                df = df.append(pd.DataFrame([row_n], columns=columns))
            else:
                df = df.append(pd.DataFrame([row], columns=columns))
            if len(df) > ws:
                full_window_flag = True

        count += 1

        if full_window_flag:
            actual_values, predicted_values = regression(df, slide, columns, actual_values, predicted_values)
            df = df.tail(-slide)
            full_window_flag = False

        if count % 1000 == 0:
            print("C: ", count)

    evaluate(od_method, actual_values, predicted_values, percentage)
