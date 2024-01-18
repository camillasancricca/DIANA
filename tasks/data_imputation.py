import pandas as pd
from kafka import KafkaConsumer, TopicPartition
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Lib import preproc_lib as pp
from Lib import dq_lib as dq
from Lib import profiling_lib as lb
from Lib import kll
from Lib import eff_apriori
from Lib.imputation.LOCF import LOCF
from Lib.imputation.rolling_mean import r_mean
from Lib.imputation.interpolation import Interpolation
#from pypots.imputation.saits import SAITS

import time
from Lib.khh import KHeavyHitters  # it uses CMS to keep track
import warnings


def evaluate(method, actual_values, predicted_values, percentage):
    actual_values = [float(item) for item in actual_values]
    predicted_values = [float(item) for item in predicted_values]
    r2 = metrics.r2_score(actual_values, predicted_values)
    print("R2-score ", r2)
    save_results(method, percentage, r2)

def save_results(method,percentage,r2):
    path = "../Risultati/Air_Quality/data_imputation_regression.csv"
    cols = ["method","percentage","r2"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[method, percentage, r2]], columns=cols)
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
    df_c.dropna(axis=0, inplace=True)
    y = df_c.pop('PM2.5')
    # X = df_c[r_cols]
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

start_time = time.time()
# collections.Iterable = collections.abc.Iterable
# collections.Mapping = collections.abc.Mapping
# collections.MutableSet = collections.abc.MutableSet
# collections.MutableMapping = collections.abc.MutableMapping


topic = 'csv-topic'

columns = ['date_time', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
           'RAIN', 'wd', 'WSPM', 'station', 'arrive_time']
types = ['string', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
         'float', 'float', 'float', 'float', 'string', 'string']

date_time_format = '%Y-%m-%d %H:%M:%S'
null_value = '  NA'
locf = LOCF()
r_mean = r_mean(types)
interp = Interpolation()
#saits = SAITS(n_steps=30,
#              n_features=97,
#              n_layers=2,
#              d_model=256,
#              d_inner=128,
#              n_head=4,
#              d_k=64,
#              d_v=64,
#              dropout=0.1,
#              epochs=10,
#              patience=3,
#              learning_rate=1e-3,
#              )

methods = ['none','LOCF','mean','interpolation']
percentage = 50
ws = 336
slide = 48
for method in methods:
    count = 0
    full_window_flag = False
    actual_values = []
    predicted_values = []

    with open('../Datasets/real_values.pkl', 'rb') as pick:
        true_values = pickle.load(pick)

    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
    tp = TopicPartition(topic, 0)
    consumer.assign([tp])
    consumer.seek_to_beginning()
    df = pd.DataFrame(columns=columns)

    p = consumer.position(tp)
    print("Connessione fatta")
    for message in consumer:
        row = str(message.value.decode('utf-8'))
        if row == 'finito':
            break

        row = pp.row_str_preprocess(row)
        row = pp.row_type_preprocess(row, columns, types, null_value)
        # row = pp.row_add_datetime(row, date_time_format)
        count += 1

        if method == 'none':
            df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
        if method == 'LOCF':
            row = locf.new_value(row)
            df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
        if method == 'mean':
            df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
            if len(df) > ws:
                column_means = df.mean()
                df = df.fillna(column_means)
        if method == 'interpolation':
            df = df.append(pd.DataFrame([row], columns=columns), ignore_index=True)
            if len(df) > ws:
                df = interp.interpolate(df, regression)

        if len(df)>ws:
            full_window_flag = True

        if count % 2000 == 0:
            print(count)

        if full_window_flag:
            actual_values, predicted_values = regression(df, slide, columns, actual_values, predicted_values)
            df = df.tail(-slide)
            full_window_flag = False

    evaluate(method, actual_values, predicted_values, percentage)


