import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental.enable_iterative_imputer import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
#import impyute as impy
#from hyperimpute.plugins.imputers import Imputers
#import utils
import fancyimpute as fi
from sklearn.preprocessing import OrdinalEncoder

from sklearn import linear_model


class no_impute:
    def __init__(self):
        self.name = 'No imputation'

    def fit(self, df):
        return df

class impute_standard:
    def __init__(self):
        self.name = 'Standard'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Missing")
        return df

class drop:
    def __init__(self):
        self.name = 'Drop'

    def fit_cols(self, df):
        df = df.dropna(axis=1, how='any')
        return df

    def fit_rows(self, df):
        df = df.dropna(axis=0, how='any')
        return df

class impute_mean:
    def __init__(self):
        self.name = 'Mean'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_std:
    def __init__(self):
        self.name = 'Std'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].std())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].std())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_mode:
    def __init__(self):
        self.name = 'Mode'

    def fit(self, df):
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_median():
    def __init__(self):
        self.name = 'Median'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].median())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_knn():
    def __init__(self):
        self.name = 'KNN'

    def fit(self, df):
        # create an object for KNNImputer
        imputer = KNNImputer(n_neighbors=5)

        df_m = pd.DataFrame(imputer.fit_transform(df))
        df_m = pd.DataFrame(df_m)
        df_m.columns = df.columns
        return df_m

    def fit_cat(self, df):
        columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        # create an object for KNNImputer
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        imputer = KNNImputer(n_neighbors=5)

        df_m = pd.DataFrame(imputer.fit_transform(df))
        df_m = pd.DataFrame(df_m)
        df_m.columns = columns

        return df_m

class impute_mice:
    def __init__(self):
        self.name = 'Mice'

    def fit(self, df):
        multivariate_impute_pipe = ColumnTransformer([
            ("impute_num", IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5),
                                            max_iter=100), df.columns)
        ]
        )
        df_knn = multivariate_impute_pipe.fit_transform(df)
        df_knn = pd.DataFrame(df_knn)
        df_knn.columns = df.columns
        return df_knn

    def fit_cat(self, df):
        columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        multivariate_impute_pipe = ColumnTransformer([
            ("impute_num", IterativeImputer(estimator=BayesianRidge(),
                                            max_iter=100), list(range(0, len(columns))))
        ]
        )
        df_knn = multivariate_impute_pipe.fit_transform(df)
        df_knn = pd.DataFrame(df_knn)
        df_knn.columns = columns
        return df_knn

## fancyimpute

class impute_matrix_factorization:
    def __init__(self):
        self.name = 'Matrix factorization'

    def fit(self, df):
        try:
            df_imp = fi.MatrixFactorization(verbose=False).fit_transform(df)
            df = pd.DataFrame(df_imp,columns=df.columns)
        except:
            print('skipped 0% missing data dataset')
        return df

    def fit_cat(self, df):
        columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        try:
            df_imp = fi.MatrixFactorization(verbose=False).fit_transform(df)
            df = pd.DataFrame(df_imp,columns=columns)
        except:
            print('skipped 0% missing data dataset')
        return df

class impute_soft:
    def __init__(self):
        self.name = 'Soft impute'

    def fit(self, df):
        try:
            df_imp = fi.SoftImpute(max_iters=100,verbose=False).fit_transform(df)
            df = pd.DataFrame(df_imp,columns=df.columns)
        except:
            print('skipped 0% missing data dataset')
        return df

    def fit_cat(self, df):
        columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        try:
            df_imp = fi.SoftImpute(max_iters=100,verbose=False).fit_transform(df)
            df = pd.DataFrame(df_imp,columns=columns)
        except:
            print('skipped 0% missing data dataset')
        return df



# metodi aggiunti da Enrico

class impute_bfill:
    def __init__(self):
        self.name = 'Backward'

    def fit(self, df):
        df = df.fillna(method="bfill")

        # if the last values are still missing I impute them with a ffill
        df = df.fillna(method="ffill")
        return df


class impute_ffill:
    def __init__(self):
        self.name = 'Forward'

    def fit(self, df):
        df = df.fillna(method="ffill")

        # if the first values are still missing I impute them with a bfill
        df = df.fillna(method="bfill")
        return df


class impute_random:
    def __init__(self):
        self.name = 'Random'

    def fit(self, df):
        for col in df.columns:
            number_missing = df[col].isnull().sum()
            observed_values = df.loc[df[col].notnull(), col]
            df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df

    def fit_single_column(self, df, col):
        number_missing = df[col].isnull().sum()
        observed_values = df.loc[df[col].notnull(), col]
        df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df


class impute_linear_regression:
    def __init__(self):
        self.name = 'Linear Regression'

    def fit(self, df, missing_columns):  # missing_columns sono le colonne con missing values. Queste verranno imputate.

        # provo a mettere encoder

        # columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        for feature in missing_columns:
            df[feature + '_imp'] = df[feature]
            random_imputer = impute_random()
            df = random_imputer.fit_single_column(df, feature + '_imp')

        for feature in missing_columns:
            #imp_data["IMP" + feature] = diabetes[feature]
            parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

            # Create a Linear Regression model to estimate the missing data
            model = linear_model.LinearRegression()
            model.fit(X=df[parameters], y=df[feature + '_imp'])
            model_predicted = model.predict(df[parameters])

            # observe that I preserve the index of the missing data from the original dataframe
            # print(feature + " successfully imputed")
            df.loc[df[feature].isnull(), feature] = model_predicted[df[feature].isnull()]

        for feature in missing_columns:
            df = df.drop(feature + '_imp', axis=1)

        return df


class impute_logistic_regression:
    def __init__(self):
        self.name = 'Logistic Regression'

    def fit(self, df, missing_columns):  # missing_columns sono le colonne con missing values. Queste verranno imputate.

        # encoder
        # columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        # controllo che le missing columns hanno effettivamente dei missing values
        missing_columns = list(missing_columns)
        missing_columns[:] = [column for column in missing_columns if df[column].isnull().values.any()]

        for feature in missing_columns:
            df[feature + '_imp'] = df[feature]
            random_imputer = impute_random()
            df = random_imputer.fit_single_column(df, feature + '_imp')

        for feature in missing_columns:
            # imp_data["IMP" + feature] = diabetes[feature]
            parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

            if df[feature].nunique() < 2:  # se c'è un solo valore in quella colonna allora applico imputation con moda
                df[feature] = df[feature].fillna(df[feature].mode()[0])
            else:
                # Create a model to estimate the missing data
                model = linear_model.LogisticRegression(max_iter=100)
                model.fit(X=df[parameters], y=df[feature + '_imp'])
                model_predicted = model.predict(df[parameters])

                # observe that I preserve the index of the missing data from the original dataframe
                # print(feature + " successfully imputed")
                df.loc[df[feature].isnull(), feature] = model_predicted[df[feature].isnull()]

        for feature in missing_columns:
            df = df.drop(feature + '_imp', axis=1)

        return df


class impute_random_forest:
    def __init__(self):
        self.name = 'Random Forest'

    def fit(self, df, missing_columns):  # missing_columns sono le colonne con missing values. Queste verranno imputate.

        # provo a mettere encoder

        # columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        for feature in missing_columns:
            df[feature + '_imp'] = df[feature]
            random_imputer = impute_random()
            df = random_imputer.fit_single_column(df, feature + '_imp')

        for feature in missing_columns:
            #imp_data["IMP" + feature] = diabetes[feature]
            parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

            model = RandomForestRegressor()
            model.fit(X=df[parameters], y=df[feature + '_imp'])
            model_predicted = model.predict(df[parameters])

            # observe that I preserve the index of the missing data from the original dataframe
            print(feature + " successfully imputed")
            df.loc[df[feature].isnull(), feature] = model_predicted[df[feature].isnull()]

        for feature in missing_columns:
            df = df.drop(feature + '_imp', axis=1)

        return df


class impute_linear_and_logistic:
    def __init__(self):
        self.name = 'Linear and Logistic Regression'

#   use this imputator to impute whole datasets with linear and logistic regression
#   for whole datasets don't use impute_linear_regression and impute_logistic_regression

    def fit(self, df, missing_columns):  # missing_columns sono le colonne con missing values. Queste verranno imputate.
        # qui il dataset dovrebbe essere non ancora encodato

        categorical_columns = list(df.select_dtypes(include=['bool', 'object']).columns)

        # encoder
        # columns = df.columns
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(df[cat])
        df[cat] = oe.transform(df[cat])

        # controllo che le missing columns hanno effettivamente dei missing values
        missing_columns = list(missing_columns)
        missing_columns[:] = [column for column in missing_columns if df[column].isnull().values.any()]

        for feature in missing_columns:
            df[feature + '_imp'] = df[feature]
            random_imputer = impute_random()
            df = random_imputer.fit_single_column(df, feature + '_imp')

        # ora imputo solo le missing_columns numeriche (non categoriche)
        numerical_missing_columns = list(set(missing_columns) - set(categorical_columns))

        for feature in numerical_missing_columns:
            #imp_data["IMP" + feature] = diabetes[feature]
            parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})

            # Create a Linear Regression model to estimate the missing data
            model = linear_model.LinearRegression()
            model.fit(X=df[parameters], y=df[feature + '_imp'])
            model_predicted = model.predict(df[parameters])

            # observe that I preserve the index of the missing data from the original dataframe
            # print(feature + " successfully imputed")
            df.loc[df[feature].isnull(), feature] = model_predicted[df[feature].isnull()]

        for feature in missing_columns:
            df = df.drop(feature + '_imp', axis=1)

        # ora qui ho il dataset con le colonne numeriche imputate e quelle categoriche no
        # print(df)
        if len(categorical_columns) > 0:
            logistic_imputator = impute_logistic_regression()
            df = logistic_imputator.fit(df, df.columns)

        return df


def impute_dataset(dataset, method):

    df = dataset.copy()  # ho aggiunto questa riga, se non funziona qualcosa forse è questo!!!
    imputated_df = pd.DataFrame()

    if method == "no_impute":
        imputator = no_impute()
        imputated_df = imputator.fit(df)
    elif method == "impute_standard":
        imputator = impute_standard()
        imputated_df = imputator.fit(df)
    elif method == "drop_rows":
        imputator = drop()
        imputated_df = imputator.fit_rows(df)
    elif method == "drop_cols":
        imputator = drop()
        imputated_df = imputator.fit_cols(df)
    elif method == "impute_mean":
        imputator = impute_mean()
        imputated_df = imputator.fit_mode(df)
    elif method == "impute_std":
        imputator = impute_std()
        imputated_df = imputator.fit_mode(df)
    elif method == "impute_mode":
        imputator = impute_mode()
        imputated_df = imputator.fit(df)
    elif method == "impute_median":
        imputator = impute_median()
        imputated_df = imputator.fit_mode(df)
    elif method == "impute_knn":
        imputator = impute_knn()
        imputated_df = imputator.fit_cat(df)
    elif method == "impute_mice":
        imputator = impute_mice()
        imputated_df = imputator.fit_cat(df)
    elif method == "impute_matrix_factorization":
        imputator = impute_matrix_factorization()
        imputated_df = imputator.fit_cat(df)
    elif method == "impute_soft":
        imputator = impute_soft()
        imputated_df = imputator.fit_cat(df)
    elif method == "impute_bfill":
        imputator = impute_bfill()
        imputated_df = imputator.fit(df)
    elif method == "impute_ffill":
        imputator = impute_ffill()
        imputated_df = imputator.fit(df)
    elif method == "impute_random":
        imputator = impute_random()
        imputated_df = imputator.fit(df)
    elif method == "impute_linear_regression":
        imputator = impute_linear_regression()
        imputated_df = imputator.fit(df, df.columns)
    elif method == "impute_logistic_regression":
        imputator = impute_logistic_regression()
        imputated_df = imputator.fit(df, df.columns)
    elif method == "impute_random_forest":
        imputator = impute_random_forest()
        imputated_df = imputator.fit(df, df.columns)
    elif method == "impute_linear_and_logistic":
        imputator = impute_linear_and_logistic()
        imputated_df = imputator.fit(df, df.columns)
    return imputated_df


def impute_dataset_no_class(df, method, class_name):

    # se è drop_rows allora si usa metodo impute_dataset()
    if method == "drop_rows":
        return impute_dataset(df, method)

    # I remove the class
    feature_cols = list(df.columns)
    feature_cols.remove(class_name)
    df_no_class = pd.DataFrame(df, columns=feature_cols)

    # print("df with no class: ")
    # print(df_no_class)

    imputated_df = pd.DataFrame()

    if method == "no_impute":
        imputator = no_impute()
        imputated_df = imputator.fit(df_no_class)
    elif method == "impute_standard":
        imputator = impute_standard()
        imputated_df = imputator.fit(df_no_class)
    elif method == "drop_cols":
        imputator = drop()
        imputated_df = imputator.fit_cols(df_no_class)
    elif method == "impute_mean":
        imputator = impute_mean()
        imputated_df = imputator.fit_mode(df_no_class)
    elif method == "impute_std":
        imputator = impute_std()
        imputated_df = imputator.fit_mode(df_no_class)
    elif method == "impute_mode":
        imputator = impute_mode()
        imputated_df = imputator.fit(df_no_class)
    elif method == "impute_median":
        imputator = impute_median()
        imputated_df = imputator.fit_mode(df_no_class)
    elif method == "impute_knn":
        imputator = impute_knn()
        imputated_df = imputator.fit_cat(df_no_class)
    elif method == "impute_mice":
        imputator = impute_mice()
        imputated_df = imputator.fit_cat(df_no_class)
    elif method == "impute_matrix_factorization":
        imputator = impute_matrix_factorization()
        imputated_df = imputator.fit_cat(df_no_class)
    elif method == "impute_soft":
        imputator = impute_soft()
        imputated_df = imputator.fit_cat(df_no_class)
    elif method == "impute_bfill":
        imputator = impute_bfill()
        imputated_df = imputator.fit(df_no_class)
    elif method == "impute_ffill":
        imputator = impute_ffill()
        imputated_df = imputator.fit(df_no_class)
    elif method == "impute_random":
        imputator = impute_random()
        imputated_df = imputator.fit(df_no_class)
    elif method == "impute_linear_regression":
        imputator = impute_linear_regression()
        imputated_df = imputator.fit(df_no_class, df_no_class.columns)
    elif method == "impute_logistic_regression":
        imputator = impute_logistic_regression()
        imputated_df = imputator.fit(df_no_class, df_no_class.columns)
    elif method == "impute_random_forest":
        imputator = impute_random_forest()
        imputated_df = imputator.fit(df_no_class, df_no_class.columns)
    elif method == "impute_linear_and_logistic":
        imputator = impute_linear_and_logistic()
        imputated_df = imputator.fit(df_no_class, df_no_class.columns)

    # I put back the class
    imputated_df[class_name] = df[class_name]

    # print("df with class again: ")
    # print(imputated_df)

    return imputated_df




def check_technique_compatibility(dataset_name, method, column_name):

    # this function checks if an imputation method is compatible with the type of the column or not.
    # if it's not compatible returns False
    # this function works for single columns

    numerical_only_methods = ["impute_mean", "impute_std", "impute_median", "impute_linear_regression"]
    categorical_only_methods = ["impute_logistic_regression"]

    dataset_path = "datasets/" + dataset_name + "/" + dataset_name + ".csv"
    dataset = pd.read_csv(dataset_path)

    if dataset[column_name].dtype == 'float64' or dataset[column_name].dtype == 'int64':
        column_type = "numerical"
    else:
        column_type = "categorical"

    # condizione che int e impute_logistic_regression va bene
    # if dataset[column_name].dtype == 'int64' and method == "impute_logistic_regression":
    #    return True

    if column_type == "numerical" and (method in categorical_only_methods):
        return False
    if column_type == "categorical" and (method in numerical_only_methods):
        return False

    return True




















