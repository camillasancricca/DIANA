#STREAM PROFILING AND DATA QUALITY ASSESSMENT ON A LOG FILE

import pandas as pd
from kafka import KafkaConsumer, TopicPartition
import numpy as np
import tqdm

# Import custom libraries
from Lib import preproc_lib as pp
from Lib import dq_lib as dq
from Lib import profiling_lib as lb
from Lib import kll
from Lib import eff_apriori

import time
from Lib.khh import KHeavyHitters  # It uses CMS to keep track
import warnings

# Ignore warnings for clarity
warnings.filterwarnings("ignore")

# Record start time for performance measurement
start_time = time.time()

# Define Kafka topic
topic = 'csv-topic'

# Define columns and their data types
columns = ['date_time', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP',
           'RAIN', 'wd', 'WSPM', 'station', 'arrive_time']
types = ['string', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
         'float', 'float', 'float', 'float', 'string', 'string']

# Initialize Kafka consumer
consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
tp = TopicPartition(topic, 0)
consumer.assign([tp])
consumer.seek_to_beginning()

# TODO: Set parameters from a configuration file

# Define null values, date_time format, and counters
null_value = [' NA', '  NA', np.NaN, 'nan', ' ', '', None, -200]
date_time_format = '%Y-%m-%d %H:%M:%S'
late_values = 0
checks = 0
violations = 0
iqr_count = 0
wrong = np.zeros(len(columns))
count = np.zeros(len(columns))
null = np.zeros(len(columns))
mean = np.zeros(len(columns))
std_dev = np.zeros(len(columns))
missing_value_rows = 0
quantile = []
top_k = []
lst = []
actual_fd = None
lb.create_files(columns)
dq.create_files()

# Initialize data structures for heavy hitters and quantiles
for i in range(len(columns)):
    top_k.append(KHeavyHitters(5))
    if types[i] == 'int' or types[i] == 'float':
        quantile.append(kll.KLL(256))
    else:
        quantile.append(None)

# Get current position in Kafka topic
p = consumer.position(tp)
print("Connection established")

# Iterate over messages from Kafka
for message in consumer:
    row = str(message.value.decode('utf-8'))
    null_row = False
    if row == 'finito':
        break

    # Preprocess the row
    row = pp.row_str_preprocess(row)
    row = pp.row_type_preprocess(row, columns, types, null_value)

    # Initialize min and max values for profiling
    for col in range(len(columns)):
        if message.offset == p:
            mini, maxi = lb.min_max(row, columns)
        count[col] += 1
        if row[col] is None:
            null[col] += 1
            null_row = True
        else:
            if row[col] < mini[col]:
                mini[col] = row[col]
            if row[col] > maxi[col]:
                maxi[col] = row[col]
    lst.append(row)
    late_values = dq.timeliness(late_values, row[-1], date_time_format)

    # Update statistics for mean and standard deviation
    std_dev = lb.c_std_dev(std_dev, mean, count, row, types)
    mean = lb.c_mean(mean, count, row, types)
    lb.c_quant(quantile, row)
    lb.c_top_k(top_k, row)
    if null_row:
        missing_value_rows += 1

    # Calculate quantiles for outlier detection
    if ((count[0] - 1) % 24) == 0:
        q_25, q_50, q_75 = lb.comp_quantiles(quantile)
    iqr = dq.compute_wrong_values(wrong, q_25, q_75, row, types)
    if iqr:
        iqr_count += 1

    # Process data every 7 days (168 hours)
    if count[0] > 1 and (count[0] % 168) == 0:
        df = pd.DataFrame(lst, columns=columns)
        lst = []
        lb.check_variation(df, mean, std_dev, count, types, columns)
        lb.save_values(df, mean, std_dev)
        dq.comp(count, null, count[0] / 168)
        dq.acc(wrong, count, null, count[0] / 168)
        dq.write_timeliness(late_values, count[0])
        #actual_fd, checks, violations = eff_apriori.rules(df, 0.25, 0.95, actual_fd, checks, violations)
        #dq.write_cons(checks, violations, count[0])
        #print("FDs: \n", actual_fd)
        # actual_fd = lb.check_fd(df, actual_fd)
        # print("C: ", count[0])
        # print("FDs: ", len(actual_fd))
        # if len(actual_fd) < 15:
        #    exit()

# Print final statistics
print("************")
print("COUNT: ", count)
print("NULL: ", null)
print("ROWS WITH AT LEAST A MISSING VALUE: ", missing_value_rows)
print("MIN: ", mini)
print("MAX: ", maxi)
print("MEAN: ", mean)
print("STD_DEV: ", np.sqrt(std_dev))
print("WRONG: ", wrong)
print("ROWS WITH AN OUTLIER (IQR)", iqr_count)

# Print quantiles and heavy hitters
lb.print_quantiles(quantile)
print()
lb.print_top_k(top_k, columns)
# print(tabulate([count, null, mini, maxi, mean, np.sqrt(std_dev)], headers=columns))

# Print elapsed time
print()
print("--- %s seconds ---" % (time.time() - start_time))

# Close Kafka consumer
# consumer.close()
