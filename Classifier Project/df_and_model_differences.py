import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn.metrics import f1_score


# This function returns the accuracy of a dataset df with respect to the true original dataset
# It considers a value correct only if it's equal to the real value
def dataset_accuracy(df, true_dataset):
    # print(df == true_dataset)
    count_equal_cells = (df == true_dataset).sum().sum()
    # print(count_equal_cells)

    accuracy = count_equal_cells / (df.shape[0]*df.shape[1])

    print(accuracy)

    return accuracy

    # numero di null
    # print((df.isnull().sum().sum()) / (df.shape[0]*df.shape[1]))


# This function returns the MSE of a dataset df with respect to the true original dataset
# It sums the MSE considering one cell at a time.
def dataset_mse(dataset, true_dataset):

    df = dataset.copy()
    true_dataset_copy = true_dataset.copy()

    # riempio i nan con 0 così se ci sono nan comunque calcola un mse abbastanza sensato
    df = df.fillna(0)
    true_dataset_copy = true_dataset_copy.fillna(0)

    sub_df = df - true_dataset_copy
    print(sub_df)
    squared_sub_df = sub_df**2
    print(squared_sub_df)
    mse = squared_sub_df.sum().sum()

    mse = mse / (df.shape[0]*df.shape[1])

    print(mse)

    return mse


# tu gli dai un modello e un dataset e lui ti calcola le feature importance di quel modello
def permutation_feature_importance(model, X, y):

    r = permutation_importance(model, X, y, n_repeats=10, random_state=0, scoring="f1_weighted")

    for i in r.importances_mean.argsort()[::-1]:
        # questo if ti prende solo le features più importanti
        # if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X.columns[i]:<8} "
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

    plt.bar(X.columns, r.importances_mean)
    plt.show()


def create_pdp(model, X, feature):

    # Create the data that we will plot
    pdp_feature1 = pdp.pdp_isolate(model=model, dataset=X, model_features=X.columns,
                                   feature=feature)

    # plot it
    pdp.pdp_plot(pdp_feature1, feature)
    plt.show()


# This function takes two models and a dataset,
# and returns how many equal predictions these two models make on the dataset
def prediction_differences(model1, model2, X):
    predictions1 = model1.predict(X)
    predictions2 = model2.predict(X)

    print(predictions1)
    print(predictions2)

    n_equal_predictions = (predictions1 == predictions2).sum()

    # print(n_equal_predictions)

    percentage_equal_predictions = n_equal_predictions / len(predictions1)

    print(percentage_equal_predictions)

    return percentage_equal_predictions





