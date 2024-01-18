import pandas as pd
import numpy as np
import dirty_compl
import data_imputation_tecniques
import algorithms_classification
import features as f
import feature_selection
import df_and_model_differences
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt


def scores_before_and_after_imputation():

    dataset_name = "iris"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    score = algorithms_classification.classification(df, df.columns[-1], "lr")
    print("score before imputation: " + str(score))

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    print(df_with_missing)

    # imputator = data_imputation_tecniques.impute_knn()
    # imputed_df = imputator.fit(df_with_missing)

    # imputation with class:
    # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_mean")
    # imputation without class:
    imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing, "impute_mean", df_with_missing.columns[-1])
    print(imputed_df)

    score = algorithms_classification.classification(imputed_df, df.columns[-1], "lr")
    print("score after imputation: " + str(score))


def plot_scores():
    imputation_methods = ["clean_dataset", "impute_mean", "impute_linear_regression", "impute_knn", "impute_mice"]
    scores_with_class = [0.5887, 0.5065, 0.6150, 0.6790, 0.6290]
    scores_without_class = [0.5887, 0.5065, 0.5023, 0.4867, 0.5293]

    plt.title("Imputation considering the class as a feature")
    plt.ylabel("ML model performance")
    plt.axhline(y=0.5887, color='r')  # horizontal line for clean dataset performances

    plt.bar(imputation_methods, scores_with_class)
    plt.show()


def check_accuracy():
    df = pd.read_csv('datasets/iris/iris.csv')

    df_list = dirty_compl.dirty(1, "iris", df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    print(df_with_missing)

    imputator = data_imputation_tecniques.impute_mice()
    imputed_df = imputator.fit(df_with_missing)
    print(imputed_df)

    df_and_model_differences.dataset_accuracy(imputed_df, df)


def check_mse():
    dataset_name = "wine"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    # imputation with class:
    imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_mice")
    # imputation without class:
    # imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing, "impute_mice", df_with_missing.columns[-1])

    print(imputed_df)

    df_and_model_differences.dataset_mse(df, imputed_df)


def plot_mse_and_performance():
    dataset_name = "wine"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    imputation_methods = ["no_impute", "impute_mean", "impute_mode", "impute_median", "impute_bfill", "impute_ffill",
                          "impute_linear_regression", "impute_knn", "impute_mice", "impute_soft"]
    ML_model_scores = []
    mse_list = []

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    for imputation_method in imputation_methods:
        # imputation with class:
        # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, imputation_method)
        # imputation without class:
        imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing,
                                                                       imputation_method,
                                                                       df_with_missing.columns[-1])
        score = algorithms_classification.classification(imputed_df, df.columns[-1], "lr")
        ML_model_scores.append(score)
        mse_value = df_and_model_differences.dataset_mse(imputed_df, df)
        mse_list.append(mse_value)

    print(ML_model_scores)
    print(mse_list)

    # now I plot the results

    # line plots
    """
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('imputation methods')
    ax1.set_ylabel('ML scores', color=color)
    ax1.plot(imputation_methods, ML_model_scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('mse', color=color)
    ax2.plot(imputation_methods, mse_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()
    """

    # bar plots

    fig, axs = plt.subplots(2)
    fig.suptitle('Wine')
    axs[0].set_xlabel('imputation methods')
    axs[0].set_ylabel('ML model performance')
    axs[0].bar(imputation_methods, ML_model_scores)
    axs[1].set_ylabel('mse')
    axs[1].bar(imputation_methods, mse_list, color='tab:red')
    plt.show()

    print("original dataset at the end (check): ")
    print(df)


def check_feature_importance():
    dataset_name = "iris"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    # model on clean dataset
    # X, y = algorithms_classification.dataset_preprocessing(df, df.columns[-1])
    # model = algorithms_classification.train_model(X, y, "lr")

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    # imputation with class:
    # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_knn")
    # imputation without class:
    imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing, "impute_mode", df_with_missing.columns[-1])

    # model trained on imputed dataset
    X_imputed, y_imputed = algorithms_classification.dataset_preprocessing(imputed_df, imputed_df.columns[-1])
    model_imputed = algorithms_classification.train_model(X_imputed, y_imputed, "lr")

    # df_and_model_differences.permutation_feature_importance(model, X, y)
    df_and_model_differences.permutation_feature_importance(model_imputed, X_imputed, y_imputed)


def check_pdp():
    dataset_name = "iris"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    # model on clean dataset
    X, y = algorithms_classification.dataset_preprocessing(df, df.columns[-1])
    model = algorithms_classification.train_model(X, y, "lr")

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    # imputation with class:
    # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_knn")
    # imputation without class:
    imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing, "impute_mode",
                                                                   df_with_missing.columns[-1])

    # model trained on imputed dataset
    X_imputed, y_imputed = algorithms_classification.dataset_preprocessing(imputed_df, imputed_df.columns[-1])
    model_imputed = algorithms_classification.train_model(X_imputed, y_imputed, "lr")

    # df_and_model_differences.create_pdp(model, X, "petal_width")
    df_and_model_differences.create_pdp(model_imputed, X_imputed, "petal_width")



def check_prediction_differences():
    dataset_name = "wine"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    X, y = algorithms_classification.dataset_preprocessing(df, df.columns[-1])

    model_clean = algorithms_classification.train_model(X, y, "lr")

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    # imputation with class:
    # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, "impute_mice")
    # imputation without class:
    imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing, "impute_mice", df_with_missing.columns[-1])

    X_imputed, y_imputed = algorithms_classification.dataset_preprocessing(imputed_df, imputed_df.columns[-1])

    model_imputed = algorithms_classification.train_model(X_imputed, y_imputed, "lr")

    df_and_model_differences.prediction_differences(model_clean, model_imputed, X)


    # voglio fare una prova
    # clean_f1_score = f1_score(y, model_clean.predict(X), average='weighted')
    # print(clean_f1_score)


def plot_prediction_differences():
    dataset_name = "wine"
    file_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"

    df = pd.read_csv(file_path)
    print(df)

    imputation_methods = ["impute_mean", "impute_linear_regression", "impute_knn", "impute_mice"]
    prediction_differences_list = []

    X, y = algorithms_classification.dataset_preprocessing(df, df.columns[-1])
    model_clean = algorithms_classification.train_model(X, y, "lr")

    df_list = dirty_compl.dirty(1, dataset_name, df.columns[-1], "uniform")
    df_with_missing = df_list[0]

    for imputation_method in imputation_methods:
        # imputation with class:
        # imputed_df = data_imputation_tecniques.impute_dataset(df_with_missing, imputation_method)
        # imputation without class:
        imputed_df = data_imputation_tecniques.impute_dataset_no_class(df_with_missing,
                                                                       imputation_method,
                                                                       df_with_missing.columns[-1])

        X_imputed, y_imputed = algorithms_classification.dataset_preprocessing(imputed_df, imputed_df.columns[-1])
        model_imputed = algorithms_classification.train_model(X_imputed, y_imputed, "lr")
        prediction_difference_value = df_and_model_differences.prediction_differences(model_clean, model_imputed, X)

        prediction_differences_list.append(prediction_difference_value)

    print(prediction_differences_list)

    # now I plot the results

    plt.title("Similarity of predictions with respect to the model trained on the clean dataset")
    plt.xlabel("Imputation methods (without class)")
    plt.ylabel("Similarity")
    plt.bar(imputation_methods, prediction_differences_list, color='green')
    plt.show()



if __name__ == '__main__':
    # scores_before_and_after_imputation()
    # plot_scores()
    # check_accuracy()
    # check_mse()
    # check_feature_importance()
    check_pdp()
    # check_prediction_differences()
    # plot_mse_and_performance()
    # plot_prediction_differences()














