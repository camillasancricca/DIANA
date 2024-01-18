import pandas as pd
import dirty_accuracy
import numpy as np
import numpy.random
import random
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import feature_selection

def feature_sel(dataframe, class_name):
    dataframe, indexes = feature_selection.feature_selection(dataframe, class_name)
    dataframe.to_csv('/Users/martinacaffagnini/Tesi/CodiceTesi/prova.csv')



def initialize(dataframe, name, class_name, s):
    dirty, resT, resV = dirty_accuracy.injection(df_pandas=dataframe, seed=s, name=name, name_class=class_name)
    for i in range(0,5):
        tmp1 = dirty[i]
        tmp2 = resT[i]
        tmp3 = resV[i]
        tmp1.to_csv('/home/cappiello/martina/datasets_dirty/'+class_name+'_Dirty'+str(5-i)+'0_'+str(s)+'.csv', index=False)
        tmp2.to_csv('/home/cappiello/martina/datasets_dirty/'+class_name+'_ResT'+str(5-i)+'0_'+str(s)+'.csv', index=False)
        tmp3.to_csv('/home/cappiello/martina/datasets_dirty/'+class_name+'_ResV'+str(5-i)+'0_'+str(s)+'.csv', index=False)

def bubbleSort(arr, index):
    n = len(arr)
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                index[j], index[j + 1] = index[j + 1], index[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return arr, index
    return arr, index

def get_actualDF(labels_list, resV):  
    actualDF = pd.DataFrame()
    for element in labels_list:
        a = len(resV[element].to_list())
        actual = []
        for i in range(0, a):
            if resV[element].to_list()[i] == 0:
                actual.append(0)
            else: 
                actual.append(1)
        actualDF[element]=actual
    return actualDF

def get_actual_hard(resT, labels_list):
    hardDF = pd.DataFrame()
    for element in labels_list:
        a = []
        lista = resT[element].to_list()
        for i in range(0,len(lista)):
            if lista[i]=='h':
                a.append(1)
            else:
                a.append(0)
        hardDF[element]=a
    return hardDF

def get_actual_medium(resT, labels_list):
    mediumDF = pd.DataFrame()
    for element in labels_list:
        a = []
        lista = resT[element].to_list()
        for i in range(0,len(lista)):
            if lista[i]=='m':
                a.append(1)
            else:
                a.append(0)
        mediumDF[element]=a
    return mediumDF

def get_actual_easy(resT, labels_list):
    easyDF = pd.DataFrame()
    for element in labels_list:
        a = []
        lista = resT[element].to_list()
        for i in range(0,len(lista)):
            if lista[i]=='e':
                a.append(1)
            else:
                a.append(0)
        easyDF[element]=a
    return easyDF

def hard_1(labels_list, actual_hard, outlier_indexes):
    found_h = pd.DataFrame()

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_hard[element].to_list():
            if e == 1: 
                if i in outlier_indexes:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        found_h[element]= actual
    return found_h

def medium_1(labels_list, actual_medium, outlier_indexes):
    found_m = pd.DataFrame()

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_medium[element].to_list():
            if e == 1: 
                if i in outlier_indexes:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        found_m[element]= actual
    return found_m

def easy_1(labels_list, actual_easy, outlier_indexes):
    found_e = pd.DataFrame()

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_easy[element].to_list():
            if e == 1: 
                if i in outlier_indexes:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        found_e[element]= actual
    return found_e

def easy_2(labels_list, actual_easy, outlier_indexes):
    found_e = pd.DataFrame()
    k = 0

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_easy[element].to_list():
            if e == 1: 
                if i in outlier_indexes[k]:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        k += 1
        found_e[element]= actual
    return found_e

def medium_2(labels_list, actual_medium, outlier_indexes):
    found_m = pd.DataFrame()
    k = 0

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_medium[element].to_list():
            if e == 1: 
                if i in outlier_indexes[k]:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        k += 1
        found_m[element]= actual
    return found_m

def hard_2(labels_list, actual_hard, outlier_indexes):
    found_h = pd.DataFrame()
    k = 0

    for element in labels_list:

        i = 0
        actual = []
        for e in actual_hard[element].to_list():
            if e == 1: 
                if i in outlier_indexes[k]:
                    actual.append(1)
                else: 
                    actual.append(0)
            else: 
                actual.append(0)
            i += 1
        k += 1
        found_h[element]= actual
    return found_h

def KNN(labels_list, dirtyL):
    lists=[]
    values=[]
    for element in labels_list:
        X = dirtyL[element].values
        nbrs = NearestNeighbors(n_neighbors = 3)
        nbrs.fit(X.reshape(-1,1))
        # distances and indexes of k-neaighbors from model outputs
        distances, indexes = nbrs.kneighbors(X.reshape(-1,1))
        # plot mean of k-distances of each observation
        #plt.plot(distances.mean(axis =1))
        outlier_index = np.where(distances.mean(axis = 1) > 0.0001 )
        # filter outlier values
        outlier_values = dirtyL[element].iloc[outlier_index]

        list_index =[]
        list_outliers=[]
    
        for x in outlier_values.index:
            list_index.append(x)
        for x in outlier_values:
            list_outliers.append(x)
        lists.append(list_index)
        values.append(list_outliers)
    return lists, values


def ZSB(threshold, labels_list, df):
    # Robust Zscore as a function of median and median
    # mean absolute deviation (MAD) defined as
    # z-score = |x – median(x)| / mad(x)
    index_list=[]
    outlier_list=[]
    for element in labels_list:
        data = df[element].values
        median = np.median(data)
        median_absolute_deviation = np.median(np.abs(data - median))
        modified_z_scores = (data - median) / median_absolute_deviation
        outliers = data[np.abs(modified_z_scores) > threshold]
        index = np.where(np.abs(modified_z_scores)>threshold)[0].tolist()
        outlier_list.append(outliers)
        index_list.append(index)
    return outlier_list, index_list

def STD(data, index):
    mean = float(data.mean())
    std = float(np.std(data))
    V1 = mean + 3 * std
    V2 = mean - 3 * std
    outliers=[]
    outliers_ind=[]
    i=0
    for x in data:
        if (x > V1) | (x<V2):
            outliers.append(x)
            outliers_ind.append(index[i])
        i+=1

    return outliers, outliers_ind

def PERC(data, index):
    V1 = numpy.percentile(data, 99)
    V2 = numpy.percentile(data, 1)
    outliers=[]
    outliers_ind=[]
    i=0
    for x in data:
        if (x > V1) | (x<V2):
            outliers.append(x)
            outliers_ind.append(index[i])
        i+=1

    return outliers, outliers_ind

def ISO(data, element):
    model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.5),max_features=1.0)
    model.fit(data[[element]])
    prova = pd.DataFrame()
    prova['scores']=model.decision_function(data[[element]])
    prova['anomaly']=model.predict(data[[element]])
    anomaly=prova.loc[prova['anomaly']==-1]
    outliers_ind=list(anomaly.index)      

    return outliers_ind

def IQR(data, index):
    bubbleSort(data, index)
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    outliers=[]
    outliers_ind=[]
    i=0
    for x in data:
        if (x > upper_range) | (x<lower_range):
            outliers.append(x)
            outliers_ind.append(index[i])
        i+=1

    return outliers, outliers_ind

def LOF(X):
    data = X
    # requires no missing value
    # select top 10 outliers
    from sklearn.neighbors import LocalOutlierFactor

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=4, contamination=0.1)

    clf.fit_predict(X)

    LOF_scores = clf.negative_outlier_factor_
    # Outliers tend to have a negative score far from -1

    #print(LOF_scores)

    outliers = X[LOF_scores < -1.1].index
    return outliers


def localOutlierFactor(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []
    
    for i in range(0,5):
        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(i+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(i+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(i+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        labels_list = dirty_accuracy.get_names(data, name)
        data = data.drop(data.columns[[4]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        actual=get_actualDF(labels_list, resV)

        for element in labels_list:
            indexes = LOF(data[[element]])
            
            tmp2 = actual[element].to_list()
            a = len(resV[element].to_list())
            
            found = dirty_accuracy.fill_listA(indexes,a)
            confusion_matrix = metrics.confusion_matrix(tmp2, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            cm_display.plot()
            prova = metrics.f1_score(tmp2, found)
    
            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_1(labels_list, actual_easy, indexes)
            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_1(labels_list, actual_medium, indexes)
            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_1(labels_list, actual_hard, indexes)
            
            fscore.append(element+'_'+str(i+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()
    results.to_csv('/home/cappiello/martina/LOF/'+name+'_fscore_LOF_'+str(s)+'.csv', index=False)
    return results



def kNearestNeighbors(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []
    for k in range(0,5):
        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(k+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(k+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(k+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index

        labels_list = dirty_accuracy.get_names(data, name)
        actual=get_actualDF(labels_list, resV)

        outlier_indexes, outlier_value = KNN(labels_list, data)

        i=0
        for element in labels_list:
            a = len(resV[element].to_list())
            found = dirty_accuracy.fill_listA(outlier_indexes[i],a )
            b = actual[element].to_list()
            confusion_matrix = metrics.confusion_matrix(b, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            cm_display.plot()
            prova = metrics.f1_score(b, found)
            
            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_2(labels_list, actual_easy, outlier_indexes)

            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_2(labels_list, actual_medium, outlier_indexes)

            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_2(labels_list, actual_hard, outlier_indexes)
            fscore.append(element+'_'+str(k+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()
            i+=1
    results.to_csv('/home/cappiello/martina/KNN/'+name+'_fscore_KNN_'+str(s)+'.csv', index=False)
    return results




def isolationForest(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []

    for i in range(0,5):
        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(i+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(i+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(i+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        labels_list = dirty_accuracy.get_names(data, name)
        data = data.drop(data.columns[[4]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        actual=get_actualDF(labels_list, resV)

        for element in labels_list:
            indexes = ISO(data, element)
            
            tmp2 = actual[element].to_list()
            a = len(resV[element].to_list())
            
            found = dirty_accuracy.fill_listA(indexes,a)
            confusion_matrix = metrics.confusion_matrix(tmp2, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            cm_display.plot()
            prova = metrics.f1_score(tmp2, found)
            
            
            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_1(labels_list, actual_easy, indexes)
            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_1(labels_list, actual_medium, indexes)
            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_1(labels_list, actual_hard, indexes)
            
            fscore.append(element+'_'+str(i+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()
    results.to_csv('/home/cappiello/martina/ISO/'+name+'_fscore_ISO_'+str(s)+'.csv', index=False)
    return results



def percentile(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []
    for i in range(0,5):
        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(i+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(i+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(i+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        labels_list = dirty_accuracy.get_names(data, name)
        actual=get_actualDF(labels_list, resV)

        for element in labels_list:

            tmp = data[element]
            tmp1 = [i for i in range(0, len(tmp))]
            outliers, indexes = PERC(tmp, tmp1)    
            tmp2 = actual[element].to_list()
            a = len(resV[element].to_list())
            
            found = dirty_accuracy.fill_listA(indexes,a)
            confusion_matrix = metrics.confusion_matrix(tmp2, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            prova = metrics.f1_score(tmp2, found)
            cm_display.plot()

            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_1(labels_list, actual_easy, indexes)

            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_1(labels_list, actual_medium, indexes)

            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_1(labels_list, actual_hard, indexes)
            fscore.append(element+'_'+str(i+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()
    results.to_csv('/home/cappiello/martina/PERC/'+name+'_fscore_PERC_'+str(s)+'.csv', index=False)
    return results




def standardDeviation(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []
    for i in range(0,5):  
        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(i+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(i+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(i+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        labels_list = dirty_accuracy.get_names(data, name)
        actual=get_actualDF(labels_list, resV)

        for element in labels_list:
            tmp = data[element]
            tmp1 = [i for i in range(0, len(tmp))]
            outliers, indexes = STD(tmp, tmp1)    
            tmp2 = actual[element].to_list()
            a = len(resV[element].to_list())
            
            found = dirty_accuracy.fill_listA(indexes,a)
            confusion_matrix = metrics.confusion_matrix(tmp2, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            prova = metrics.f1_score(tmp2, found)
            
            cm_display.plot()
            
            
            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_1(labels_list, actual_easy, indexes)
            
            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_1(labels_list, actual_medium, indexes)
           
            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_1(labels_list, actual_hard, indexes)
            fscore.append(element+'_'+str(i+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()

            
    results.to_csv('/home/cappiello/martina/STD/'+name+'_fscore_STD_'+str(s)+'.csv', index=False)
    return results




def zModifiedScore(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []
    for k in range(0,5):

        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(k+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(k+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(k+1)+'0_'+str(s)+'.csv')
        
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index

        labels_list = dirty_accuracy.get_names(data, name)
        actual=get_actualDF(labels_list, resV)
        outlier_value, outlier_indexes = ZSB(2, labels_list, data)

        i=0
        for element in labels_list:
            a = len(resV[element].to_list())
            found = dirty_accuracy.fill_listA(outlier_indexes[i],a )
            b = actual[element].to_list()
            confusion_matrix = metrics.confusion_matrix(b, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            prova = metrics.f1_score(b, found)
            cm_display.plot()

            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_2(labels_list, actual_easy, outlier_indexes)

            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_2(labels_list, actual_medium, outlier_indexes)

            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_2(labels_list, actual_hard, outlier_indexes)
            fscore.append(element+'_'+str(k+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()

            i+=1
    results.to_csv('/home/cappiello/martina/ZSB/'+name+'_fscore_ZSB_'+str(s)+'.csv', index=False)
    return results



def interquartileRange(name, s):
    results = pd.DataFrame()
    results['column']=''
    results['f1-score']=''
    results['easy']=''
    results['medium']=''
    results['hard']=''
    fscore = []

    for i in range(0,5):

        data = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_Dirty'+str(i+1)+'0_'+str(s)+'.csv')
        resV = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResV'+str(i+1)+'0_'+str(s)+'.csv')
        resT = pd.read_csv('/home/cappiello/martina/datasets_dirty/'+name+'_ResT'+str(i+1)+'0_'+str(s)+'.csv')
        #data = data.drop(data.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resV = resV.drop(resV.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        #resT = resT.drop(resT.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
        labels_list = dirty_accuracy.get_names(data, name)
        actual=get_actualDF(labels_list, resV)
        

        for element in labels_list:
            tmp = data[element].to_list()
            tmp1 = [i for i in range(0, len(tmp))]
            outliers, indexes = IQR(tmp, tmp1)
            tmp2 = actual[element].to_list()
            a = len(resV[element].to_list())
            
            found = dirty_accuracy.fill_listA(indexes,a)
            confusion_matrix = metrics.confusion_matrix(tmp2, found)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Not Outliers', 'Outliers'])
            
            prova = metrics.f1_score(tmp2, found)

            cm_display.plot()
            
            actual_easy = get_actual_easy(resT, labels_list)
            found_e = easy_1(labels_list, actual_easy, indexes)

            actual_medium = get_actual_medium(resT, labels_list)
            found_m = medium_1(labels_list, actual_medium, indexes)

            actual_hard = get_actual_hard(resT, labels_list)
            found_h = hard_1(labels_list, actual_hard, indexes)
            fscore.append(element+'_'+str(i+1)+'0')
            fscore.append(prova)
            fscore.append(metrics.accuracy_score(actual_easy, found_e))
            fscore.append(metrics.accuracy_score(actual_medium, found_m))
            fscore.append(metrics.accuracy_score(actual_hard, found_h))

            results.loc[len(results.index)] = fscore
            fscore=[]
            plt.close()
        
    results.to_csv('/home/cappiello/martina/IQR/'+name+'_fscore_IQR_'+str(s)+'.csv', index=False)
    return results


