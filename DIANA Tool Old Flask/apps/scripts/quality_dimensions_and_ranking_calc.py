import numpy as np
import pandas as pd
import apps.scripts.kb_test as kb

def accuracy_value(df, profile, columns, typeNUMlist, min_values, max_values):
    # Syntactic Accuracy: Number of correct values/total number of values
    i=0
    correct_values_tot=0
    tot_n_values = profile['table']['n']*len(columns)
    for var in columns:
        if var in typeNUMlist:
            range_correct_values = np.arange(float(min_values[i]),float(max_values[i])) #start value, stop value and step value. I choose a small step as we do not know the real content of the columns
            correct_values_i=sum(1 for item in df[var] if item in range_correct_values)
            correct_values_tot = correct_values_tot + correct_values_i
            i+=1
    accuracy = correct_values_tot / tot_n_values * 100
    return accuracy

def uniqueness_value(profile):
    # Uniqueness = percentage calculated as Cardinality (count of the number 
    # of distinct actual values) divided by the total number of records.
    n_tot_distinct=0
    tot_n_values=0

    for var in profile['variables'].values():
        n_tot_distinct+=var['n_distinct']
        tot_n_values+=var['n']

    uniqueness=n_tot_distinct/tot_n_values*100
    return uniqueness

def completeness_value(profile, columns):
    # Completeness: Number of not null values/total number of values
    tot_n_values = profile['table']['n']*len(columns)
    tot_not_null = tot_n_values - profile['table']['n_cells_missing']
    completeness = tot_not_null / tot_n_values *100
    return completeness

def rank_kb(df, algorithm):
    kb_read_example = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    ranking_kb = kb.predict_ranking(kb_read_example, df, algorithm)
    ranking_kb = str(ranking_kb)
    ranking_kb = ranking_kb.replace("[", "")
    ranking_kb = ranking_kb.replace("]", "")
    ranking_kb = ranking_kb.replace("'", "")
    ranking_kb = ranking_kb.split()
    return ranking_kb

def rank_dim(accuracy, uniqueness, completeness):
    ordered_values = sorted([accuracy, uniqueness, completeness], reverse=True)
    ranking_dim = []
    for i in range(3):
        if ordered_values[i] == accuracy:
            ranking_dim.append('ACCURACY')
        if ordered_values[i] == completeness:
            ranking_dim.append('COMPLETENESS')
        if ordered_values[i] == uniqueness:
            ranking_dim.append('UNIQUENESS')
    return ranking_dim

def average_ranking(ranking_kb, ranking_dim):
    # Get the unique values in both lists using set() function
    print("kb  ")
    print(ranking_kb)
    print("dim  ")
    print(ranking_dim)
    accuracy=0
    completeness=0 
    uniqueness = 0
    for i in range(3):
        if ranking_kb[i] == 'ACCURACY':
            if i==0: accuracy = accuracy + 0.5 * 60
            if i==1: accuracy = accuracy +  0.5 * 30
            if i==2: accuracy = accuracy +  0.5 * 10
            
        if ranking_kb[i] == 'COMPLETENESS':
            if i==0: completeness = completeness +  0.5 * 60
            if i==1: completeness = completeness +  0.5 * 30
            if i==2: completeness = completeness +  0.5 * 10
        if ranking_kb[i] == 'UNIQUENESS':
            if i==0: uniqueness = uniqueness +  0.5 * 60
            if i==1: uniqueness = uniqueness +  0.5 * 30
            if i==2: uniqueness = uniqueness +  0.5 * 10
        

    for i in range(3):
        if ranking_dim[i] == 'ACCURACY':
            if i==0: accuracy = accuracy +  0.5 * 60
            if i==1: accuracy = accuracy +  0.5 * 30
            if i==2: accuracy = accuracy +  0.5 * 10
                   
        if ranking_dim[i] == 'COMPLETENESS':
            if i==0: completeness = completeness +  0.5 * 60
            if i==1: completeness = completeness +  0.5 * 30
            if i==2: completeness = completeness +  0.5 * 10
        
        if ranking_dim[i] == 'UNIQUENESS':
            if i==0: uniqueness = uniqueness +  0.5 * 60
            if i==1: uniqueness = uniqueness +  0.5 * 30
            if i==2: uniqueness = uniqueness +  0.5 * 10
    

    sort = sorted([accuracy, uniqueness, completeness], reverse=True)
    ranking=[]
    for i in range(3):
        if sort[i] == accuracy:
            ranking.append('ACCURACY')
        if sort[i] == completeness:
            ranking.append('COMPLETENESS')
        if sort[i] == uniqueness:
            ranking.append('UNIQUENESS')
    # Print the new list with the average order
    return ranking