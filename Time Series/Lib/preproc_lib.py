import datetime
import numpy as np


def row_str_preprocess(row):
    row = row.replace("]", "")
    row = row.replace("[", "")
    row = row.replace("'", "")
    row = row.replace('"', '')
    row = row.replace(" ", "")
    row = row.split(",")
    return row


def row_type_preprocess(row, columns, types, null_value):
    for col in range(len(columns)-1):
        if row[col] in null_value or len(row[col]) == 0 or row[col] == np.nan:
            row[col] = None
            continue
        elif types[col] == "int":
            row[col] = int(float(row[col]))
        elif types[col] == "int" or types[col] == "float":
            row[col] = float(row[col])
    return row


def row_add_datetime(row, format):
    year = row[1]
    month = row[2]
    day = row[3]
    hour = row[4]
    dt = datetime.datetime(year, month, day, hour)
    formatted_dt = dt.strftime(format)
    row.append(formatted_dt)
    return row