import numpy as np
from scipy.integrate import simps

def best_method(imp_scores):

    areas = []

    def compute_area(scores):
        area_es = 0
        for i in range(0,len(scores)-1):
            a = scores[i]
            b = scores[i+1]
            y = [a,b]
            area_es += simps(y, dx=10)
        return area_es

    for s in imp_scores:
        areas.append(compute_area(s))

    areas_sorted = areas.copy()
    areas_sorted.sort(reverse=True)

    best_area = areas_sorted[0]

    return find_method(best_area, areas, [])

def best_method_perline(scores_tot):

    max_tot = []

    for scores in scores_tot:

        s = scores[0]
        max_index = 0
        max = scores[0][0]
        max_all = []

        for i in range(0,len(s)):
            for j in range(0,len(scores)):
                if scores[j][i] > max:
                    max = scores[j][i]
                    max_index = j
            max_all.append(find_method_n(max_index))
            max_index = 0
            if i == len(s)-1:
                break
            max = scores[0][i+1]

        max_tot.append(max_all)
    return max_tot

def best_tree_methods_perdata(scores):

    max_tot = []

    for s in scores:

        scores_a = s.copy()
        scores_a = np.array(scores_a)
        index_a = scores_a.argsort(axis=0)
        scores_a.sort(axis=0)
        sorted_scores = [scores_a, index_a]
        sorted_scores = np.flip(sorted_scores, axis=1)
        max_tot.append([find_method_n(sorted_scores[1][0]),find_method_n(sorted_scores[1][1]),find_method_n(sorted_scores[1][2])])

        print(str(sorted_scores[0][0]-sorted_scores[0][2]))

    return max_tot

def best_tree_methods_perline(scores_tot):

    max_tot = []

    for scores in scores_tot:

        s = scores[0]
        max_all = []

        for i in range(0, len(s)):
            scores_a = []
            for j in range(0, len(scores)):
                scores_a.append(scores[j][i])

            scores_a = np.array(scores_a)
            index_a = scores_a.argsort(axis=0)
            scores_a.sort(axis=0)
            sorted_scores = [scores_a, index_a]
            sorted_scores = np.flip(sorted_scores, axis=1)
            max_all.append([find_method_n(sorted_scores[1][0]),find_method_n(sorted_scores[1][1]),find_method_n(sorted_scores[1][2])])
            if i == len(s)-1:
                break

        max_tot.append(max_all)
    return max_tot

def best_tree_methods_perline_with_scores(scores_tot):

    max_tot = []

    for scores in scores_tot:

        s = scores[0]
        max_all = []

        for i in range(0, len(s)):
            scores_a = []
            for j in range(0, len(scores)):
                scores_a.append(scores[j][i])

            scores_a = np.array(scores_a)
            index_a = scores_a.argsort(axis=0)
            scores_a.sort(axis=0)
            sorted_scores = [scores_a, index_a]
            sorted_scores = np.flip(sorted_scores, axis=1)
            max_all.append([find_method_n(sorted_scores[1][0]),sorted_scores[0][0],find_method_n(sorted_scores[1][1]),sorted_scores[0][1],find_method_n(sorted_scores[1][2]),sorted_scores[0][2]])
            if i == len(s)-1:
                break

        max_tot.append(max_all)
    return max_tot

def three_best_methods(imp_scores):

    areas = []

    def compute_area(scores):
        area_es = 0
        for i in range(0,len(scores)-1):
            a = scores[i]
            b = scores[i+1]
            y = [a,b]
            area_es += simps(y, dx=10)
        return area_es

    for s in imp_scores:
        areas.append(compute_area(s))

    base = areas[0]
    for i in range(0,len(areas)):
        areas[i] = areas[i]/base

    areas_sorted = areas.copy()
    areas_sorted.sort(reverse=True)

    first = areas_sorted[0]
    second = areas_sorted[1]
    third = areas_sorted[2]

    first_method = find_method(first,areas,[])
    second_method = find_method(second,areas,[first_method])
    third_method = find_method(third,areas,[first_method,second_method])

    return first_method,round((first-1)*100,4),second_method,round((second-1)*100,4),third_method,round((third-1)*100,4)

def find_method(value, areas, esclude):
    if value == areas[0]:
        if ("NOP" not in esclude):
            return "NOP"
    if value == areas[1]:
        if ("DROP_COLS" not in esclude):
            return "DROP_COLS"
    if value == areas[2]:
        if ("DROP_ROWS" not in esclude):
            return "DROP_ROWS"
    if value == areas[3]:
        if ("STANDARD" not in esclude):
            return "STANDARD"
    if value == areas[4]:
        if ("MEAN" not in esclude):
            return "MEAN"
    if value == areas[5]:
        if ("MEDIAN" not in esclude):
            return "MEDIAN"
    if value == areas[6]:
        if ("STD" not in esclude):
            return "STD"
    if value == areas[7]:
        if ("MODE" not in esclude):
            return "MODE"
    if value == areas[8]:
        if ("KNN" not in esclude):
            return "KNN"
    if value == areas[9]:
        if ("MICE" not in esclude):
            return "MICE"
    if value == areas[10]:
        if ("MF" not in esclude):
            return "MF"
    else:
        return "SVD"

def find_method_n(n):
    if n == 0:
            return "NOP"
    if n == 1:
            return "DROP_COLS"
    if n == 2:
            return "DROP_ROWS"
    if n == 3:
            return "STANDARD"
    if n == 4:
            return "MEAN"
    if n == 5:
            return "MEDIAN"
    if n == 6:
            return "STD"
    if n == 7:
            return "MODE"
    if n == 8:
            return "KNN"
    if n == 9:
            return "MICE"
    if n == 10:
            return "MF"
    else:
        return "SVD"
