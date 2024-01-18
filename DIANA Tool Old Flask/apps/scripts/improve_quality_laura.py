import random as rd
import numpy as np

# imputing missing values

def imputing_missing_values(dataset):

    for col in dataset.columns:
        if (dataset[col].dtype != "object"):
            dataset[col] = dataset[col].fillna(dataset[col].mean())
        else:
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    return dataset

# delete missing values

def delete_missing_values_rows(dataset):
    dataset = dataset.dropna(axis=0, how='any')
    return dataset

def delete_missing_values_cols(dataset):
    dataset = dataset.dropna(axis=1, how='any')
    return dataset

# outlier correction
#esempio di range: ranges = [[0,1], [0,1]]
def outlier_correction(dataset, outlier_range):

    for col in dataset.columns:
        index = dataset.columns.get_loc(col)

        if (dataset[col].dtype != "object"):
            # if ((dataset[col].mean() < outlier_range[index][0]) | (dataset[col].mean() > outlier_range[index][1])):
                dataset.loc[((dataset[col] < outlier_range[index][0]) | (dataset[col] > outlier_range[index][1])) & dataset[col].notnull(),col]=np.nan
            # else:
            #     dataset.loc[((dataset[col] < outlier_range[index][0]) | (dataset[col] > outlier_range[index][1])) & dataset[col].notnull(),col]=dataset[col].mean()
        # else:
        #     dataset.loc[~dataset[col].isin(outlier_range[index]) & dataset[col].notnull(),col]=dataset[col].mode()[0]
        #     dataset.loc[~dataset[col].isin(outlier_range[index]) & dataset[col].notnull(), col] = np.nan



        #chiama i metodi di imputation !!

    return dataset

def remove_duplicates(df):
    return df.drop_duplicates()
