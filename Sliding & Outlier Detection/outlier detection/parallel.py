import pandas as pd
import dirty_accuracy
import features 
import numpy as np
import best_method as bm
from multiprocessing import Pool
from itertools import repeat
import dirty_accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def generate_seed(n_seed,n_elements):
    seed = []
    seeds = []
    for r in range(0,n_seed):
        for i in range(0,n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds

def parallel_exec(name, class_name):
    #sempre multipli
    n_instances_tot = 8
    n_parallel_jobs = 8
    n_instances_x_job = int(n_instances_tot/n_parallel_jobs)
    iqr = []
    iso=[]
    perc=[]
    std=[]
    zsb=[]
    knn=[]
    lof=[]
    iqr_df= pd.DataFrame()
    iso_df= pd.DataFrame()
    perc_df= pd.DataFrame()
    std_df= pd.DataFrame()
    zsb_df= pd.DataFrame()
    knn_df= pd.DataFrame()
    lof_df= pd.DataFrame()

    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    itr = zip(repeat(name), repeat(class_name), seed)
    with Pool(processes=n_parallel_jobs) as pool:
        results = pool.starmap(procedure, itr)
        #In che modo vengono ritornati i res di procedure?
        print(results)

        results_final = []


        #results_final = np.array([decision_trees,knn,naive_bayes,logistic_r])

        for i in range(0,n_parallel_jobs):
            iqr.append(results[i][0])
            iso.append(results[i][1])
            perc.append(results[i][2])
            std.append(results[i][3])
            zsb.append(results[i][4])
            knn.append(results[i][5])
            lof.append(results[i][6])

        for i in range(0,len(iqr)):     
            iqr_df = iqr_df + iqr[i]
            iso_df = iso_df + iso[i]
            perc_df = perc_df + perc[i]
            std_df = std_df + std[i]
            zsb_df = zsb_df + zsb+[i]
            knn_df = knn_df + knn[i]
            lof_df = lof_df + lof[i]
            iqr_df['column']=iqr[i]['column']
            iso_df['column']=iso[i]['column']
            perc_df['column']=perc[i]['column']
            std_df['column']=std[i]['column']
            zsb_df['column']=zsb[i]['column']
            knn_df['column']=knn[i]['column']
            lof_df['column']=lof[i]['column']

        iqr_df['f1-score']=iqr_df['f1-score']/n_instances_tot
        iqr_df['easy']=iqr_df['easy']/n_instances_tot
        iqr_df['medium']=iqr_df['medium']/n_instances_tot
        iqr_df['hard']=iqr_df['hard']/n_instances_tot
        results_final.append(iqr_df)

        iso_df['f1-score']=iso_df['f1-score']/n_instances_tot
        iso_df['easy']=iso_df['easy']/n_instances_tot
        iso_df['medium']=iso_df['medium']/n_instances_tot
        iso_df['hard']=iso_df['hard']/n_instances_tot  
        results_final.append(iso_df)

        perc_df['f1-score']=perc_df['f1-score']/n_instances_tot
        perc_df['easy']=perc_df['easy']/n_instances_tot
        perc_df['medium']=perc_df['medium']/n_instances_tot
        perc_df['hard']=perc_df['hard']/n_instances_tot   
        results_final.append(perc_df)

        std_df['f1-score']=std_df['f1-score']/n_instances_tot
        std_df['easy']=std_df['easy']/n_instances_tot
        std_df['medium']=std_df['medium']/n_instances_tot
        std_df['hard']=std_df['hard']/n_instances_tot   
        results_final.append(std_df)

        zsb_df['f1-score']=zsb_df['f1-score']/n_instances_tot
        zsb_df['easy']=zsb_df['easy']/n_instances_tot
        zsb_df['medium']=zsb_df['medium']/n_instances_tot
        zsb_df['hard']=zsb_df['hard']/n_instances_tot   
        results_final.append(zsb_df)

        knn_df['f1-score']=knn_df['f1-score']/n_instances_tot
        knn_df['easy']=knn_df['easy']/n_instances_tot
        knn_df['medium']=knn_df['medium']/n_instances_tot
        knn_df['hard']=knn_df['hard']/n_instances_tot   
        results_final.append(knn_df)

        lof_df['f1-score']=lof_df['f1-score']/n_instances_tot
        lof_df['easy']=lof_df['easy']/n_instances_tot
        lof_df['medium']=lof_df['medium']/n_instances_tot
        lof_df['hard']=lof_df['hard']/n_instances_tot     
        results_final.append(lof_df)
        return results_final

def procedure(name, class_name, seed):

    res = []
    df = pd.read_csv('/home/cappiello/martina/datasets/'+class_name+'FS.csv')
    functions.initialize(df, name, class_name, seed)
    
    iqr = functions.interquartileRange(class_name, seed)    
    res.append(iqr)
    iso = functions.isolationForest(class_name, seed)
    res.append(iso)
    perc = functions.percentile(class_name, seed)
    res.append(perc)
    std = functions.standardDeviation(class_name, seed)
    res.append(std)
    zsb = functions.zModifiedScore(class_name, seed)
    res.append(zsb)
    knn = functions.kNearestNeighbors(class_name, seed)
    res.append(knn)
    lof = functions.localOutlierFactor(class_name, seed)
    res.append(lof)
    return res

    
    
    
if __name__ == '__main__':

    #name = ['letter', 'oil', 'qualityred', 'qualitywhite', 'ecoli', 'frogs', 'acustic', 'cancer']
    #class_name = ['letter', 'oil', 'qualityred', 'qualitywhite', 'ecoli', 'frogs', 'acustic', 'cancer']
    name = ['letter']
    #parametri che vi servono in ingresso alla procedure

    for n in name: 
        [iqr, iso, perc, std, zsb, knn, lof]  = parallel_exec(n, n) #prende results final per scriverlo su csv

        results = pd.DataFrame()
        results['column']=''
        results['f1-score']=''
        results['easy']=''
        results['medium']=''
        results['hard']=''
        iqr['type'] = ['IQR']*len(iqr)
        iso['type'] = ['ISO']*len(iso)
        perc['type'] = ['PERC']*len(perc)
        std['type'] = ['STD']*len(std)
        zsb['type'] = ['ZSB']*len(zsb)
        knn['type'] = ['KNN']*len(knn)
        lof['type'] = ['LOF']*len(lof)

        total = pd.concat([iqr, iso, perc, std, knn, lof], ignore_index=True)

        for element in total['column'].unique():
            tmp = total.loc[total['column'] == element]
            ne = int(tmp[['easy']].idxmax())
            nm = int(tmp[['medium']].idxmax())
            nh = int(tmp[['hard']].idxmax())
            nt = int(tmp[['f1-score']].idxmax())
            new_row = [element, total.loc[nt, 'type'], total.loc[nt, 'f1-score'], 
                        total.loc[ne, 'type'],  total.loc[ne, 'easy'],
                        total.loc[nm, 'type'], total.loc[nm, 'medium'],
                        total.loc[nh, 'type'], total.loc[nh, 'hard']]
            results.loc[len(results)]= new_row

        results.to_csv('/home/cappiello/martina/datasets/'+name+'.csv', index=False)
