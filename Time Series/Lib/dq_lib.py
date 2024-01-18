from datetime import datetime


def create_files(name=""):
    path = "../Risultati/comp" + name + ".txt"
    f = open(path, 'w')
    f.close()
    f = open("../Risultati/acc.txt", 'w')
    f.close()
    f = open("../Risultati/cons.txt", 'w')
    f.close()
    f = open("../Risultati/time.txt", 'w')
    f.close()


def comp(count, null, weeks):
    f = open("../Risultati/comp.txt", 'a')
    null_v = null.sum()
    count_v = count.sum()
    perc = (1 - null_v / count_v) * 100
    s = "After {} weeks, there was a completeness of {}% \n".format(weeks, perc)
    f.write(s)
    f.close()


def comp_sw(count, null, date):
    f = open("../Risultati/compsw.txt", 'a')
    perc = (1 - null / count) * 100
    s = "{} : In the last 2 days, there was a completeness of {}% \n".format(date, perc)
    f.write(s)
    f.close()


def compute_wrong_values(wrong, q_25, q_75, row, types):
    wrong_rows_iqr = 0
    iqr_out = False
    for i in range(len(row)):
        if types[i] == "int" or types[i] == "float":
            l_bound = q_25[i] - 1.5 * (q_75[i] - q_25[i])
            u_bound = q_75[i] + 1.5 * (q_75[i] - q_25[i])
            if row[i] is not None and (row[i] < l_bound or row[i] > u_bound):
                wrong[i] += 1
                iqr_out = True
    return iqr_out


def acc(wrong, count, null, weeks):
    f = open("../Risultati/acc.txt", 'a')
    null_v = null.sum()
    count_v = count.sum()
    wrong_v = wrong.sum()
    not_null_v = count_v - null_v
    perc = (1 - wrong_v / not_null_v) * 100
    s = "After {} weeks, there was a accuracy of {}% \n".format(weeks, perc)
    f.write(s)
    f.close()
    return None

# def cons():


def timeliness(late_values, arrive_time, date_time_format): #TODO Mettere formula con currency e timeliness
    now = datetime.now()
    arrive_time = datetime.strptime(arrive_time, date_time_format)
    delay = now - arrive_time
    if delay.total_seconds() > 2:
        late_values += 1
    return late_values


def write_timeliness(late_values, count):
    f = open("../Risultati/time.txt", 'a')
    perc = late_values/count * 100
    s = "After {} values, there was {}% of late values\n".format(count, perc)
    f.write(s)
    f.close()
    return None


def write_cons(checks, violations, count):
    f = open("../Risultati/cons.txt", 'a')
    if checks > 0:
        perc = violations / checks * 100
    else:
        perc = 0.0
    s = "After {} values, there was {}% of FDs violations\n".format(count, perc)
    f.write(s)
    f.close()
    return None