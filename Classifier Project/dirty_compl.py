import random
import pandas as pd
import numpy as np


def dirty(seed, name, name_class, method): # name_class sarebbe il nome della colonna che contiene le class labels. Così li non mettono i missing values
    if method == "uniform":
        return uniform(seed, name, name_class)
    elif method == "column":
        return columns(seed, name, name_class)
    else:
        return rows(seed, name, name_class)


def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return  df


def uniform(seed, name, name_class):

    np.random.seed(seed)

    #%%

    file_path = "datasets/" + name + "/" + name + ".csv"
    df_pandas = pd.read_csv(file_path)
    df_list = []


    perc = [0.50, 0.40, 0.30, 0.20, 0.10]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p,1-p]
        df_dirt = check_datatypes(df_dirt)

        for col in df_dirt.columns:

            if col!=name_class:

                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True,col]=np.nan

        df_list.append(df_dirt)
        # print("saved {}-completeness{}%".format(name, round((1-p)*100)))
    return df_list  # ti ritorna una lista di dataset con gli errori dentro


def columns(seed, name, name_class):

    np.random.seed(seed)

    #%%

    file_path = "datasets/" + name + "/" + name + ".csv"
    df_pandas = pd.read_csv(file_path)
    df_list = []

    columns = df_pandas.columns
    n_columns = len(columns)
    n_columns = int(round(n_columns/2,0))
    p = 0.50
    columns = columns.drop(name_class)

    for n in range(n_columns,-1,-1):
        if n == 0:
            break
        dirty_columns = random.sample(list(columns),k=n)
        df_dirt = df_pandas.copy()
        comp = [p,1-p]
        df_dirt = check_datatypes(df_dirt)

        for col in columns:

            if col!=name_class and col in dirty_columns:

                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True,col]=np.nan

        df_list.append(df_dirt)
        print("saved {}-completeness-{}cols".format(name, n))
    return df_list


def rows(seed, name, name_class):

    np.random.seed(seed)

    # %%

    file_path = "datasets/" + name + "/" + name + ".csv"
    df_pandas = pd.read_csv(file_path)
    df_list = []

    tot_rows = len(df_pandas)
    perc_rows = 0.10
    n_rows = int(round(tot_rows/2,0))
    p = 0.50
    columns = df_pandas.columns
    n_columns = int(round(len(columns)/2,0))

    df_pandas = df_pandas.reset_index()  # make sure indexes pair with number of rows

    gap = int(round(n_rows*0.1))

    columns = columns.drop(name_class)

    for r in range(n_rows,-1,-gap):
        if r == 0:
            break
        dirty_rows = random.sample(range(0,tot_rows-1), k=r)
        df_dirt = df_pandas.copy()
        df_dirt = check_datatypes(df_dirt)

        for index, row in df_dirt.iterrows():
            if int(index) in dirty_rows:

                dirty_columns = random.sample(list(columns),k=n_columns)

                for col in df_dirt.columns:
                    if col != name_class and col in dirty_columns:

                        df_dirt.loc[int(index), col] = np.nan

        df_list.append(df_dirt.drop(columns=["index"]))
        print("saved {}-completeness-{}rows".format(name, r))
    return df_list




# aggiunto da Enrico

def dirty_single_column(seed, dataset, column_name, name_class):
    # il metodo usato è solo uniform
    # qui gli passo proprio il dataset, forse è meglio

    np.random.seed(seed)

    # %%

    df_pandas = dataset[[column_name]].copy()
    df_list = []

    perc = [0.50, 0.40, 0.30, 0.20, 0.10]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p, 1 - p]
        df_dirt = check_datatypes(df_dirt)

        for col in df_dirt.columns:

            if col != name_class:
                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True, col] = np.nan

        # potrei fare qua il cambio da column a dataset
        df_dirt_complete = dataset.copy()
        df_dirt_complete[column_name] = df_dirt[column_name]
        df_list.append(df_dirt_complete)
        # print("saved {}-completeness{}%".format(column_name, round((1 - p) * 100)))

    # print(df_list)
    return df_list
























