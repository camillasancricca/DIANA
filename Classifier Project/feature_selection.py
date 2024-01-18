import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def feature_selection(df, class_name):

    if df.shape[1] <= 5:
        return df  # ho già 5 o meno colonne

    cat = list(df.select_dtypes(include=['bool', 'object']).columns)
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    oe.fit(df[cat])
    df[cat] = oe.transform(df[cat])

    feature_cols = list(df.columns)
    feature_cols.remove(class_name)

    X = df[1:][feature_cols]  # Features
    y = df[1:][class_name]  # Target variable

    feature_columns = list(X.columns)

    X = np.nan_to_num(X)
    X = pd.DataFrame(X, columns=feature_columns)

    forest = RandomForestClassifier()
    forest.fit(X,y)

    importances = forest.feature_importances_

    # print(importances)

    indices = np.argsort(importances)[::-1]  # questo ti ritorna già ordine decrescente

    feature_columns = X.columns
    selected_columns = feature_columns[indices]
    importances = importances[indices]

    # print(indices)
    # print(importances)
    # print(selected_columns)

    selected_columns = selected_columns[:4]
    # print(selected_columns)

    df_new = pd.DataFrame(df, columns=selected_columns)
    df_new[class_name] = df[class_name]

    # print(df)
    # print(df_new)
    return df_new













