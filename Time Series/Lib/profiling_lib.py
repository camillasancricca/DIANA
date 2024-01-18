import numpy as np

from Lib import fdtool
from csv import writer


def min_max(row, columns):
    mini = []
    maxi = []
    for i in range(len(columns)):
        mini.append(row[i])
        maxi.append(row[i])
    return mini, maxi


def c_mean(mean, count, row, types):
    for col in range(len(types)):
        if (types[col] == 'int' or types[col] == 'float') and (row[col] is not None and row[col] is not np.nan):
            mean[col] = (mean[col] * (count[col] - 1) + row[col]) / count[col]

    return mean


def c_std_dev(std_dev, mean, count, row, types):
    for col in range(len(types)):
        if (types[col] == 'int' or types[col] == 'float') and row[col] is not None:
            if count[col] < 2:
                a = 1
            else:
                a = (count[col] - 2) / (count[col] - 1)
            std_dev[col] = a * std_dev[col] + (1 / count[col]) * ((row[col] - mean[col]) ** 2)

    return std_dev


def c_quant(quantile, row):
    for col in range(len(row)):
        if quantile[col] is not None and row[col] is not None:
            quantile[col].update(row[col])


def c_top_k(top_k, row):
    for col in range(len(row)):
        if row[col] is not None:
            top_k[col].add(row[col])


def comp_quantiles(quantile):
    q_25 = []
    q_50 = []
    q_75 = []
    for col in range(len(quantile)):
        if quantile[col] is None:
            q_25.append(-1)
            q_50.append(-1)
            q_75.append(-1)
        else:
            a = list(zip(*quantile[col].cdf()))
            b = list(filter(lambda i: i > 0.25, a[1]))[0]
            idx = a[1].index(b)
            q_25.append(a[0][idx])
            a = list(zip(*quantile[col].cdf()))
            b = list(filter(lambda i: i > 0.5, a[1]))[0]
            idx = a[1].index(b)
            q_50.append(a[0][idx])
            a = list(zip(*quantile[col].cdf()))
            b = list(filter(lambda i: i > 0.75, a[1]))[0]
            idx = a[1].index(b)
            q_75.append(a[0][idx])

    return q_25, q_50, q_75


def print_quantiles(quantile):
    perc = [0.25, 0.50, 0.75]
    for p in perc:
        print()
        print(f"{(p * 100)}%")
        for col in range(len(quantile)):
            if quantile[col] is None:
                print("---", end="\t")
            else:
                a = list(zip(*quantile[col].cdf()))
                b = list(filter(lambda i: i > p, a[1]))[0]
                idx = a[1].index(b)
                print(a[0][idx], end="\t")


def print_top_k(top_k, col):
    for i in range(len(top_k)):
        elements = top_k[i].k()
        print("Top-5", col[i], "are: ")
        for elem in elements:
            print(elem, " - ", top_k[i].query(elem))





def check_fd(df, actual_fd):
    fds = fdtool.main(actual_fd, df)
    return fds


def check_variation(df, mean, std_dev, count, types, columns):
    s = None
    f = open("../Risultati/Anomaly_values.txt", 'a')
    for col in range(len(columns)):
        if types[col] == "int" or types[col] == "float":
            win_mean = df[columns[col]].describe()[1]
            if win_mean < mean[col] * 0.5 or win_mean > mean[col] * 1.5:
                s = "WARNING: Anomaly values in : {} after {} elements \n".format(str(columns[col]), str(count[col]))
                f.write(s)
    f.close()

#def check_variation_sw(df, mean, count, types, columns):
#CUSUM ?


def create_files(columns, name=""):
    c = columns.copy()
    c.insert(0, "date_coll")
    path = "../Risultati/mean" + name + ".csv"
    f = open(path, 'w')
    w = writer(f)
    w.writerow(c)
    f.close()

    path = "../Risultati/std_dev" + name + ".csv"
    f = open(path, 'w')
    w = writer(f)
    w.writerow(c)
    f.close()

    f = open('../Risultati/Anomaly_values.txt', 'w')
    f.close()

    f = open('../Risultati/FD_eff_a_priori.txt','w')
    f.close()


def save_values(df, mean, std_dev, name=""):
    m = mean.tolist()
    sd = std_dev.tolist()
    m.insert(0, df.iloc[-1]["date_time"])
    sd.insert(0, df.iloc[-1]["date_time"])
    path = "../Risultati/mean" + name + ".csv"
    with open(path, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(m)
        f_object.close()

    path = "../Risultati/std_dev" + name + ".csv"
    with open(path, 'a', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(sd)
        f_object.close()
