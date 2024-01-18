#TEST OUTLIER DETECTION METHODS
import pandas as pd
import pickle
from kafka import KafkaConsumer, TopicPartition
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Lib import preproc_lib as pp
from Lib.outlier_det.MAD import MAD
from Lib.outlier_det.LOF import LOF
from Lib.outlier_det.z_score import z_score
from Lib.outlier_det.IForest import isoforest
from Lib.outlier_det.hst import HST
from Lib import dq_lib as dq
from Lib import profiling_lib as lb
from Lib import kll
from Lib import eff_apriori
from pyod.models.copod import COPOD

import time
from Lib.khh import KHeavyHitters  # it uses CMS to keep track
import warnings


def evaluate(method, true, predicted, percentage):

    common_values = np.intersect1d(true, predicted)
    if len(common_values) > 0:
        # Compute precision
        precision = len(common_values) / len(predicted)

        # Compute recall
        recall = len(common_values) / len(true)

        f1 = (2 * precision * recall) / (precision + recall)

    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    print("Common values: ", len(common_values))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score ", f1)
    save_results(method, percentage, precision, recall, f1)


def save_results(method, percentage, precision, recall, f1):
    path = "../Risultati/NEWeather/outlier_detection.csv"
    cols = ["method", "percentage", "precision", "recall", "f1"]
    df = pd.read_csv(path)
    row = pd.DataFrame([[method, percentage, precision, recall, f1]], columns=cols)
    df = df.append(row)

    df.to_csv(path, index=False)
    return None


warnings.filterwarnings("ignore")

scaler = StandardScaler()
start_time = time.time()
# collections.Iterable = collections.abc.Iterable
# collections.Mapping = collections.abc.Mapping
# collections.MutableSet = collections.abc.MutableSet
# collections.MutableMapping = collections.abc.MutableMapping

with open('../Datasets/outliers_index_1.pkl', 'rb') as pick:
    true_outliers = pickle.load(pick)

topic = 'csv-topic-1'
null_value = [' NA', '  NA', np.NaN, 'nan', ' ', '', None, 'NA']
date_time_format = '%Y-%m-%d %H:%M:%S'
ml_method = 'regression'
percentage = 10
methods = ['hst']

if ml_method == 'classification':
    columns = ['temp', 'dew_pnt', 'sea_lvl_press', 'visibility', 'avg_wind_spd', 'max_sustained_wind_spd', 'max_temp',
               'min_temp', 'rain', 'arrive_time']
    types = ['float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'string', 'string']
    target_cols = ['rain']
    not_target_cols = [i for i in range(9) if i not in [len(columns) - 1, len(columns) - 2]]
if ml_method == 'regression':
    columns = ['date_time', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
               'RAIN', 'wd', 'WSPM', 'station', 'arrive_time']
    types = ['string', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
             'float', 'float', 'float', 'float', 'string', 'string']
    target_cols = ['PM2.5']
    not_target_cols = [i for i in range(15) if i not in [0, 1, len(columns) - 1, len(columns) - 2]]


for method in methods:
    z = z_score(types)
    lof = LOF(n=5)
    iforest = isoforest()
    hst = HST()

    # mad = []
    # quantile = []
    # for i in range(len(columns)):
    #    quantile.append(kll.KLL(256))
    #    mad.append(MAD())
    count = 0
    c_outlier = 0
    outliers = []
    max_out = 0

    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
    tp = TopicPartition(topic, 0)
    consumer.assign([tp])
    consumer.seek_to_beginning()
    if method == 'lof':
        df = pd.DataFrame(columns=columns[1:-1])
    elif method == 'arima':
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
        if method == 'z':
            outlier = z.add_sample(row)
            if outlier == True:
                outliers.append(count)
        # lb.c_quant(quantile, row)
        # _, q_50, _ = lb.comp_quantiles(quantile)
        # for col in range(len(columns)):
        #    if row[col] is not None and (types[col] == 'int' or types[col] == 'float'):
        #        o = mad[col].add_sample(q_50[col], row[col])
        #        if o is True:
        #            outlier = True
        if method == 'lof':
            window = 1008
            row = np.array(row[1:-1]).reshape(1, -1)
            scaler = scaler.partial_fit(row)
            row = scaler.transform(row)
            df = df.append(pd.DataFrame(row, columns=columns[1:-1]))
            if len(df) > window:
                f_outliers = lof.compute(df, count, window)
                outliers = np.union1d(outliers, f_outliers)
                df = df.tail(-144)

        if method == 'iforest':
            window = 1008
            df = df.append(pd.DataFrame([row], columns=columns))
            if count == 144:
                iforest.fit(df)
            if len(df) > window:
                f_outliers = iforest.predict(df, count, window)
                outliers = np.union1d(outliers, f_outliers)
                df = df.tail(-144)

        if method == 'hst':
            window = 1008
            df = pd.DataFrame([row], columns=columns)
            o = hst.add_sample(df)
            if o:
                outliers.append(count)

        count += 1

        if count % 1000 == 0:
            print("O: ", len(outliers))
            # print("Outliers: ", outliers)
            print("C: ", count)

        # if len(df) > 120:
        #    df = df.tail(-1)
        # if count % 30 == 0:
        # model.fit(df.to_numpy())
        # print(model.labels_)
        # print(model.decision_scores_)
        # print(model.decision_scores_[model.decision_scores_ > 0.5].shape)

    evaluate(method, true_outliers, outliers, percentage)
    print("Outliers: ", len(outliers))

z.print_mean_std_dev()
