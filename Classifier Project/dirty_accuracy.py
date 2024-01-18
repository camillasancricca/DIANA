import numpy as np
import random

def out_of_range(minimum, maximum):
    foo = ["up", "down"]
    f = random.choice(foo)
    dist = maximum - minimum

    if f == "up":
        number = random.uniform(maximum, maximum + dist*10)
    else:
        number = random.uniform(minimum - dist*10, minimum)

    return number

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return  df

def injection(df_pandas, seed, name, name_class):

    np.random.seed(seed)

    #%%

    df_list = []

    #percentuale di errori
    perc = [0.50, 0.40, 0.30, 0.20, 0.10]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p,1-p]
        df_dirt = check_datatypes(df_dirt)

        for col in df_dirt.columns:
            #per evitare di cancellare le class
            if col!=name_class:

                if (df_dirt[col].dtype != "object"):
                    minimum = float(df_dirt[col].min())
                    maximum = float(df_dirt[col].max())
                    rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)
                    selected = df_dirt.loc[rand == True,col]
                    t=0
                    for i in selected:
                        selected.iloc[t:t+1] = out_of_range(minimum, maximum)
                        t+=1

                    df_dirt.loc[rand == True,col]=selected

        df_list.append(df_dirt)
        print("saved {}-accuracy{}%".format(name, round((1-p)*100)))
    return df_list
