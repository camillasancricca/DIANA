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
from sklearn.preprocessing import OrdinalEncoder

class no_impute:
    def __init__(self):
        self.name = 'No imputation'

    def fit(self, df):
        return df

class impute_standard():
    def __init__(self):
        self.name = 'Standard'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Missing")
        return df

class drop(): #using the other
    def __init__(self):
        self.name = 'Drop'

    def fit_cols(self, df):
        df = df.dropna(axis=1, how='any')
        return df

    def fit_rows(self, df):
        df = df.dropna(axis=0, how='any')
        return df

class impute_mean():
    def __init__(self):
        self.name = 'Mean'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
        return df

    def fit_mode(self, df):  #directly using this one as fi does the same thing
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_std():
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

class impute_mode():
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

class impute_mice():
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
