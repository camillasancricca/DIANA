import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pathlib
import csv
from kafka import KafkaProducer
from ksqldb import KSQLdbClient
from datetime import datetime
import time

#client = KSQLdbClient('http://localhost:8088')
#client.ksql("CREATE STREAM prova (Date VARCHAR, Postcode INT, Price INT, PropertyType VARCHAR, Bedrooms INT) WITH (kafka_topic = 'csv_topic', value_format = 'DELIMITED', timestamp = 'Date', timestamp_format='yyyy-MM-dd HH:MM:SS');")


producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
path = "Datasets/NEWeather_injected_mix_1.csv"
mex = "finito"
with open(path, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=",")
    next(csvreader, None)
    for row in csvreader:
        row = str(row).strip(",")
        row = row.replace("']", ',')
        row = row + (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        row = row + '\']'
        producer.send('csv-topic-11', bytes(row, 'utf-8'))
        #time.sleep(0.002)

producer.send('csv-topic-11', bytes(str(mex), 'utf-8'))
producer.close()

#path = "..\\Datasets\\AirQualityUCI.json"
#with open(path, 'r') as jsonfile:
#    for line in jsonfile:
#        producer.send('json-topic', bytes(line, 'utf-8'))

