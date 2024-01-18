from cgitb import small
import os
import webbrowser

import sections as sections
import streamlit as st
import time
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
import streamlit_nested_layout
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import missingno as msno
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_extras.no_default_selectbox import selectbox
from streamlit import components
import uuid
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
import math
from neo4j import GraphDatabase
from joblib import load
from sklearn import linear_model
import apps.scripts.improve_quality_laura as improve
import apps.scripts.data_imputation_tecniques as imputes



st.set_page_config(page_title="Cleaning", layout="wide", initial_sidebar_state="collapsed")

df = st.session_state['df']
if 'my_dataframe' not in st.session_state:
    st.session_state.my_dataframe = df
if 'once' not in st.session_state:
    st.session_state.once = False


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:6em;
    width:6em;
}
</style>""", unsafe_allow_html=True)

e = st.markdown("""
<style>
div[data-testid="stExpander"] div[role="button"] p {
    border: 1px solid white;
    border-radius: 5px;
    padding: 10px;
    font-size: 2rem;
    color: grey; 
    font-family: 'Verdana'
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;  /* Cambia il colore dello sfondo qui */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<style>h1{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

def accuracy_value(df):
    # Syntactic Accuracy: Number of correct values/total number of values
    correct_values_tot = 0
    tot_n_values = len(df) * len(df.columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for var in numeric_cols:
        # Accedi agli intervalli dalla variabile di stato della sessione
        min_val, max_val = st.session_state.intervals.get(var, (float('-inf'), float('inf')))

        # Calcola l'accuratezza per la colonna corrente
        correct_values_i = sum(1 for item in df[var] if not pd.isna(item) and min_val <= item <= max_val)
        correct_values_tot += correct_values_i

    accuracy = correct_values_tot / tot_n_values * 100
    return accuracy

def completeness_value(df):
    completeness = (df.isnull().sum().sum()) / (df.columns.size * len(df))
    completeness_percentage = 100 - completeness
    return completeness_percentage

consistency_value=0


def correlations(df):
    p_corr = 0

    num = list(df.select_dtypes(include=['int64','float64']).columns)

    corr = df[num].corr()

    if len(num) != 0:
        for c in corr.columns:
            a = (corr[c] > 0.7).sum() - 1
            if a > 0:
                p_corr += 1

        p_corr = p_corr / len(corr.columns)

        return round(p_corr,4),round(corr.replace(1,0).max().max(),4),round(corr.min().min(),4)

    else:
        return np.nan,np.nan,np.nan

def import_features(df):

    num = len(list(df.select_dtypes(include=['int64','float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool','object']).columns))

    rows = df.shape[0]

    cols = df.shape[1]

    corr = correlations(df)

    return rows,cols,round(num/cols,2),round(cat/cols,2),round(df.duplicated().sum()/rows,4),df.memory_usage().sum(),\
           round(df.nunique().mean()/rows,4),round(df.nunique().max()/rows,4),round(df.nunique().min()/rows,4),\
           corr[0],corr[1],corr[2]

def import_data_features(name, df, ML):

    with open(name+"_features.csv", "w") as file:
        file.write("name,tuples,features,p_num_var,p_cat_var,p_duplicates,total_size,"+
                   "p_avg_dist,p_max_dist,p_min_dist,"+
                   "p_corr,max_pears,min_pears"+
                   "\n")

        features = str(import_features(df))
        features = features.replace("(", "")
        features = features.replace(")", "")
        features = features.replace(" ", "")
        file.write(name+","+features+"\n")

    file_path = name+"_features.csv"
    data_features = pd.read_csv(file_path)

    # with this method I calculate the features of the dataset for the imputation classifier
    create_classifier_features(name, df, ML)

    return data_features

def create_testdata(dataset, ML):

    test = import_data_features("dataset", dataset, ML)
    test = test.drop(columns="name")
    return test

def predict_ranking(KB,dataset_test,ML):

    #DecisionTree
    #KNN
    #NaiveBayes

    test = create_testdata(dataset_test, ML)
    class_name = "RANKING"

    train = KB[(KB["ML"] == ML)].drop(columns=["ML","name"])

    feature_cols = list(train.columns)



    feature_cols.remove(class_name)

    X = train[1:][feature_cols] # Features
    y = train[1:][class_name] # Target variable

    feature_columns = list(X.columns)

    X = StandardScaler().fit_transform(X)

    X = np.nan_to_num(X)
    X = pd.DataFrame(X, columns=feature_columns)

    test = np.nan_to_num(test)
    test = StandardScaler().fit_transform(test)
    test = pd.DataFrame(test, columns=feature_columns)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf = clf.fit(X, y)
    pred = clf.predict(test)
    #pred_proba = clf.predict_proba(test)

    print(pred)
    #print(pred_proba)
    return pred


def density(df):
    den_tot = []

    for attr in df.columns:

        n_distinct = df[attr].nunique()
        prob_attr = []
        den_attr = 0

        for item in df[attr].unique():
            p_attr = len(df[df[attr] == item])/len(df)
            prob_attr.append(p_attr)

        avg_den_attr = 1/n_distinct

        for p in prob_attr:
            den_attr += math.sqrt((p - avg_den_attr) ** 2)
            den_attr = den_attr/n_distinct

        den_tot.append(den_attr*100)

    return den_tot


def entropy(df):
    en_tot = []

    for attr in df.columns:

        prob_attr = []

        for item in df[attr].unique():
            p_attr = len(df[df[attr] == item])/len(df)
            prob_attr.append(p_attr)

        en_attr = 0

        if 0 in prob_attr:
            prob_attr.remove(0)

        for p in prob_attr:
            en_attr += p*np.log(p)
        en_attr = -en_attr

        en_tot.append(en_attr)

    return en_tot


def import_classifier_features(df):

    num = len(list(df.select_dtypes(include=['int64', 'float64']).columns))
    cat = len(list(df.select_dtypes(include=['bool', 'object']).columns))

    rows = df.shape[0]
    cols = df.shape[1]

    # qua stava den e en = 0 e stava commentato density e entropy
    # den = [0]#density(df)
    # en = [0]#entropy(df)
    den = density(df)
    en = entropy(df)
    corr = correlations(df)

    return rows, cols, round(num / cols, 2), round(cat / cols, 2), round(df.duplicated().sum() / rows,
                                                                         4), df.memory_usage().sum(), \
        round(df.nunique().mean() / rows, 4), round(df.nunique().max() / rows, 4), round(df.nunique().min() / rows, 4), \
        round(sum(den) / len(den), 4), round(max(den), 4), round(min(den), 4), \
        round(sum(en) / len(en), 4), round(max(en), 4), round(min(en), 4), \
        corr[0], corr[1], corr[2]


def create_classifier_features(name, df, ML_model):

    file = open("dataset_classifier_features.csv", "w")

    file.write("name,n_tuples,n_attributes,p_num_var,p_cat_var,p_duplicates,total_size," +
               "p_avg_distinct,p_max_distinct,p_min_distinct," +
               "avg_density,max_density,min_density," +
               "avg_entropy,max_entropy,min_entropy," +
               "p_correlated_features,max_pearson,min_pearson," +
               "ML_ALGORITHM" +
               "\n")

    features = str(import_classifier_features(df))
    features = features.replace("(", "")
    features = features.replace(")", "")
    features = features.replace(" ", "")

    file.write(name + "," + features + "," + ML_model.lower() + "\n")

    file.close()


def rank_kb(df, algorithm):
    kb_read_example = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    ranking_kb = predict_ranking(kb_read_example, df, algorithm)
    ranking_kb = str(ranking_kb)
    ranking_kb = ranking_kb.replace("[", "")
    ranking_kb = ranking_kb.replace("]", "")
    ranking_kb = ranking_kb.replace("'", "")
    ranking_kb = ranking_kb.split()
    return ranking_kb


def rank_dim(accuracy, consistency, completeness):
    ordered_values = sorted([accuracy, consistency, completeness], reverse=True)
    ranking_dim = []
    for i in range(3):
        if ordered_values[i] == accuracy:
            ranking_dim.append('ACCURACY')
        if ordered_values[i] == completeness:
            ranking_dim.append('COMPLETENESS')
        if ordered_values[i] == consistency:
            ranking_dim.append('CONSISTENCY')
    return ranking_dim

def rank_kb(df, algorithm):
    kb_read_example = pd.read_csv('/Users/stefanoiachini/PycharmProjects/Thesis_Stefano/dataset/kb-toy-example.csv')
    ranking_kb = predict_ranking(kb_read_example, df, algorithm)
    ranking_kb = str(ranking_kb)
    ranking_kb = ranking_kb.replace("[", "")
    ranking_kb = ranking_kb.replace("]", "")
    ranking_kb = ranking_kb.replace("'", "")
    ranking_kb = ranking_kb.split()
    return ranking_kb

def average_ranking(ranking_kb, ranking_dim):
    # Get the unique values in both lists using set() function
    print("kb  ")
    print(ranking_kb)
    print("dim  ")
    print(ranking_dim)
    accuracy = 0
    completeness = 0
    consistency = 0
    for i in range(3):
        if ranking_kb[i] == 'ACCURACY':
            if i == 0: accuracy = accuracy + 0.5 * 60
            if i == 1: accuracy = accuracy + 0.5 * 30
            if i == 2: accuracy = accuracy + 0.5 * 10

        if ranking_kb[i] == 'COMPLETENESS':
            if i == 0: completeness = completeness + 0.5 * 60
            if i == 1: completeness = completeness + 0.5 * 30
            if i == 2: completeness = completeness + 0.5 * 10
        if ranking_kb[i] == 'CONSISTENCY':
            if i == 0: consistency = consistency + 0.5 * 60
            if i == 1: consistency = consistency + 0.5 * 30
            if i == 2: consistency = consistency + 0.5 * 10

    for i in range(3):
        if ranking_dim[i] == 'ACCURACY':
            if i == 0: accuracy = accuracy + 0.5 * 60
            if i == 1: accuracy = accuracy + 0.5 * 30
            if i == 2: accuracy = accuracy + 0.5 * 10

        if ranking_dim[i] == 'COMPLETENESS':
            if i == 0: completeness = completeness + 0.5 * 60
            if i == 1: completeness = completeness + 0.5 * 30
            if i == 2: completeness = completeness + 0.5 * 10

        if ranking_dim[i] == 'CONSISTENCY':
            if i == 0: consistency = consistency + 0.5 * 60
            if i == 1: consistency = consistency + 0.5 * 30
            if i == 2: consistency = consistency + 0.5 * 10

    sort = sorted([accuracy, consistency, completeness], reverse=True)
    ranking = []
    for i in range(3):
        if sort[i] == accuracy:
            ranking.append('ACCURACY')
            accuracy=0
        if sort[i] == completeness:
            ranking.append('COMPLETENESS')
            completeness=0
        if sort[i] == consistency:
            ranking.append('CONSISTENCY')
            consistency=0
    # Print the new list with the average order
    return ranking


# if __name__ == '__main__':

    # data = pd.read_csv("apps/datasets/iris.csv")
    # kb = pd.read_csv("apps/scripts/kb-toy-example.csv", sep=",")
    # string = str(predict_ranking(kb, data, "DT"))
    # string = string.replace("[", "")
    # string = string.replace("]", "")
    # string = string.replace("'", "")
    # string = string.split()
    # print(string)

def prova_db():
    URI = "neo4j://localhost"
    AUTH = ("neo4j", "ciaociao")

    driver = GraphDatabase.driver(URI, auth=AUTH)

    driver.verify_connectivity()

    dimensions = ["Consistency", "Completeness", "Accuracy"]

    # Lista per memorizzare gli output
    output_list = []

    for dimension in dimensions:
        records, _, _ = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[a:AFFECTS]->(d:DQ_DIMENSION) "
            "WHERE a.influence_type = $influence_type AND d.name = $dimension_name "
            "RETURN n.name AS name",
            influence_type="Improvement",
            dimension_name=dimension,
            database_="neo4j",
        )

        for tech in records:
            records_m, _, _ = driver.execute_query(
                "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) "
                "WHERE n.name = $technique_name "
                "RETURN m.name AS name",
                technique_name=tech["name"],
                database_="neo4j",
            )

            for meth in records_m:
                output = {"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": dimension.upper()}
                output_list.append(output)

    records, _, _ = driver.execute_query(
        "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:BENEFITS_FROM]-(ml:ML_APPLICATION) "
        "WHERE ml.application_method = $ml_algorithm "
        "RETURN DISTINCT n.name AS name",
        ml_algorithm="Logistic Regression",
        database_="neo4j"
    )

    for tech in records:
        records_m, _, _ = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) "
            "WHERE n.name = $technique_name "
            "RETURN m.name AS name",
            technique_name=tech["name"],
            database_="neo4j",
        )

        for meth in records_m:
            output = {"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": "ML_ORIENTED_ACTIONS"}
            output_list.append(output)

    driver.close()

    return output_list

def prova_classifier():
    trained_model = load('/Users/stefanoiachini/PycharmProjects/Thesis_Stefano/dataset/trained_classifier.joblib')

    dataset = pd.read_csv("/Users/stefanoiachini/PycharmProjects/Thesis_Stefano/dataset/dataset_classifier_features.csv") #vengono generate con la funzione in kb_test create_classifier_features

    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])

    ml_columns = ["ML_ALGORITHM_dt", "ML_ALGORITHM_lr", "ML_ALGORITHM_knn", "ML_ALGORITHM_nb"]
    missing_cols = set(ml_columns) - set(dataset.columns)

    for c in missing_cols:
        dataset[c] = 0

    # feature_cols = list(dataset.columns)
    feature_cols = ['n_tuples', 'n_attributes', 'p_num_var', 'p_cat_var', 'p_duplicates',
                    'total_size', 'p_avg_distinct', 'p_max_distinct', 'p_min_distinct',
                    'avg_density', 'max_density', 'min_density', 'avg_entropy', 'max_entropy',
                    'min_entropy', 'p_correlated_features', 'max_pearson', 'min_pearson',
                    'ML_ALGORITHM_dt', 'ML_ALGORITHM_knn', 'ML_ALGORITHM_lr', 'ML_ALGORITHM_nb']

    dataset = dataset.fillna(0)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    # feature_cols.remove("name")

    dataset = dataset[0:][feature_cols]  # Features

    # faccio scaling

    scaler = load('/Users/stefanoiachini/PycharmProjects/Thesis_Stefano/dataset/trained_scaler.joblib')
    # scaler = StandardScaler()
    # scaler = RobustScaler()

    dataset = scaler.transform(dataset)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    best_technique_predicted = trained_model.predict(dataset)

    print(best_technique_predicted)
    print(best_technique_predicted[0])
    return best_technique_predicted


def save_and_apply(tech, df, outlier_range):

        if tech == "remove_duplicates" or tech == "Remove Identical Duplicates" or tech == "Blocking" or tech == "Sorted Neighborhood":
            df = improve.remove_duplicates(df)
            return df


        elif tech == "impute_standard" or tech == "Standard Value Imputation" or tech == "Imputation - Standard Value Imputation" :
            # df = improve.imputing_missing_values(df)
            imputator = imputes.impute_standard()
            df = imputator.fit(df)
            return df


        elif tech == "drop_rows":
            impute = imputes.drop()
            df = impute.fit_rows(df)
            return df


        elif tech == "impute_mean" or tech == "Mean Imputation" or tech == "Imputation - Mean Imputation":
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)
            return df


        elif tech == "impute_std" or tech == "Std Imputation" or tech== "Imputation - Std Imputation":
            impute = imputes.impute_std()
            df = impute.fit_mode(df)
            return df


        elif tech == "impute_mode" or tech == "Mode Imputation" or tech == "Imputation - Mode Imputation" :
            impute = imputes.impute_mode()
            df = impute.fit(df)
            return df


        elif tech == "impute_median" or tech == "Median Imputation" or tech == "Imputation - Median Imputation":
            impute = imputes.impute_median()
            df = impute.fit_mode(df)
            return df


        elif tech == "impute_knn" or tech == "KNN Imputation" or tech == "Imputation - KNN Imputation":
            impute = imputes.impute_knn()
            df = impute.fit_cat(df)
            # df=df
            return df



        elif tech == "impute_mice" or tech == "Mice Imputation" or tech == "Imputation - Mice Imputation":
            impute = imputes.impute_mice()
            df = impute.fit_cat(df)
            # df = df
            return df


        elif tech == "Random Imputation":
            impute = imputes.impute_random()
            df = impute.fit(df)
            # df = df
            return df


        elif tech == "No Impute" or tech == "no_impute"  or tech == "Imputation - No Imputation":
            impute = imputes.no_impute()
            df = impute.fit(df)
            # df = df
            return df


        elif tech == "Linear and Logistic Imputation" or tech == "Linear Regression Imputation" or tech == "Logistic Regression Imputation":
            impute = imputes.impute_linear_and_logistic()
            df = impute.fit(df, df.columns)
            # df = df
            return df


        elif tech == "outlier_correction":
            df = improve.outlier_correction(df, outlier_range)
            return df


        elif tech == "oc_impute_standard" or tech == "Outliers Standard Value Imputation" or tech == "Outlier Correction - Outliers Standard Value Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_standard()
            df = impute.fit(df)
            return df


        elif tech == "oc_drop_rows" or tech == "Drop Outliers' Rows" or tech == "Outlier Correction - Drop Outliers' Rows":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.drop()
            df = impute.fit_rows(df)
            return df


        elif tech == "oc_impute_mean" or tech == "Outliers Mean Imputation" or tech == "Outlier Correction - Outliers Mean Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)
            return df


        elif tech == "oc_impute_std":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_std()
            df = impute.fit_mode(df)
            return df


        elif tech == "oc_impute_mode" or tech == "Outliers Mode Imputation" or tech=="Outlier Correction - Outliers Mode Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mode()
            df = impute.fit(df)
            return df


        elif tech == "oc_impute_median":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_median()
            df = impute.fit_mode(df)
            return df


        elif tech == "oc_impute_knn" or tech == "Outliers KNN Imputation" or tech == "Outlier Correction - Outliers KNN Imputation":
            df = improve.outlier_correction(df, outlier_range)
            # impute = imputes.impute_knn()
            # df = impute.fit(df)

            return df


        elif tech == "oc_impute_mice":
            df = improve.outlier_correction(df, outlier_range)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            return df

        elif tech == "z-score":
            df = improve.z_score_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            return df

        elif tech == "Min-Max":
            df = improve.min_max_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            return df

        elif tech == "Robust Scaling":
            df = improve.robust_scaler_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            return df




# imputing missing values

selected_techniques = st.session_state.setdefault("selected_techniques", [])
if "data_preparation_pipeline" not in st.session_state:
    st.session_state["data_preparation_pipeline"] = []
if "prova" not in st.session_state:
    st.session_state["prova"]=[]


outlier_range = st.session_state.intervals
num_actions = len(st.session_state["data_preparation_pipeline"])


pages = {
    "Dataset": "pagina_1",
    "Automatic Cleaning": "pagina_2",
    "Supported Cleaning": "pagina_2"
}

# Crea un menu a tendina nella barra laterale
scelta_pagina = st.sidebar.selectbox("Select a page:", list(pages.keys()))



if scelta_pagina == "Dataset":
    st.subheader("Dataset")
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.write(st.session_state.my_dataframe)

    selected_ml_technique = st.session_state.setdefault("selected_ml_technique", "KNN")

    # Aggiungi la st.radio per la selezione della tecnica di ML
    selected_ml_technique = st.selectbox("Select a ML technique:", ["KNN", "DecisionTree", "NaiveBayes"],
                                     index=["KNN", "DecisionTree", "NaiveBayes"].index(selected_ml_technique))

    # Aggiorna la variabile di stato con la tecnica di ML selezionata
    st.session_state.selected_ml_technique = selected_ml_technique
    st.write(st.session_state.selected_ml_technique)
    st.write("---")


elif scelta_pagina == "Automatic Cleaning":
    df = st.session_state['df']
    dfCol = st.session_state['dfCol']
    profile = st.session_state['profile']
    report = st.session_state['report']
    st.session_state['y'] = 0
    st.session_state['widget'] = 500

    st.title("Automatic")
    slate = st.empty()
    body = slate.container()


    def clean2():
        slate.empty()
        st.session_state['y'] = 2
        st.session_state['toBeProfiled'] = True
        # st.experimental_rerun()


    def clean3():
        slate.empty()
        st.session_state['y'] = 3
        st.session_state['toBeProfiled'] = True
        # st.experimental_rerun()


    NoDupKey = st.session_state['widget']

    correlations = profile.description_set["correlations"]
    phik_df = correlations["phi_k"]

    ind = 1
    correlationList = []
    for col in phik_df.columns:
        if ind < (len(phik_df.columns) - 1):
            for y in range(ind, len(phik_df.columns)):
                x = float(phik_df[col][y]) * 100
                if x > 85:
                    correlationList.append([col, str(phik_df.columns[y]), x])
            ind += 1

    correlationSum = {}
    for y in range(0, len(phik_df.columns)):
        x = 0
        z = 0
        for column in phik_df.columns:
            z = float(phik_df[column][y])
            x += z
        correlationSum.update({str(phik_df.columns[y]): x})

    colNum = len(st.session_state.my_dataframe.columns)
    threshold = round(0.4 * colNum)  # if a value has 40% of the attribute = NaN it's available for dropping
    nullList = st.session_state.my_dataframe.isnull().sum(axis=1).tolist()
    nullToDelete = []
    dfToDelete = st.session_state.my_dataframe.copy()
    rangeList = list(range(len(nullList)))
    for i in range(len(nullList)):
        if nullList[i] >= threshold:
            nullToDelete.append(i)
    if len(nullToDelete) > 0:
        notNullList = [i for i in rangeList if i not in nullToDelete]
        percentageNullRows = len(nullToDelete) / len(st.session_state.my_dataframe.index) * 100

    droppedList = []


    with body:
        if st.session_state['y'] == 0:
            st.subheader("Original dataset preview")
            st.dataframe(st.session_state.my_dataframe.head(50))
            st.markdown("---")
            st.write(
                "Click the button to perform automatically all the actions that the system finds suitable for your dataset, later you'll have the possibility to check the preview of the new dataset and to rollback action by action.")
            st.write(st.session_state['y'])
            if st.button("Go!"):
                st.session_state['y'] == 1
                box = st.empty()
                dfAutomatic = st.session_state.my_dataframe.copy()
                st.subheader("Original dataset preview")
                st.dataframe(st.session_state.my_dataframe.head(50))
                st.markdown("---")
                if len(nullToDelete) > 0:
                    stringDropAutomaticLoad = "Dropping the " + str(len(nullToDelete)) + " rows (" + str(
                        "%0.2f" % (percentageNullRows)) + "%) that have at least " + str(
                        threshold) + " null values out of " + str(len(st.session_state.my_dataframe.columns))
                    stringDropRollback = "Check to rollback the drop of " + str(len(nullToDelete)) + " incomplete rows"
                    stringDropAutomaticConfirmed = f"Successfully dropped **{str(len(nullToDelete))}** **rows** (" + str(
                        "%0.2f" % (percentageNullRows)) + "%) that had at least " + str(
                        threshold) + " null values out of " + str(len(st.session_state.my_dataframe.columns))
                    # dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
                    droppedList.append(["rows", nullToDelete])
                    if st.session_state['once'] == True:
                        with st.spinner(text=stringDropAutomaticLoad):
                            time.sleep(0.5)
                    st.success(stringDropAutomaticConfirmed)
                    with st.expander("Why I did it?"):
                        st.write(
                            "Incomplete rows are one of the principal sources of poor information. Even by applying the imputing technique within these rows would just be almost the same as incresing the dataset's size with non-real samples.")
                        if st.checkbox(stringDropRollback, value=False, key=len(nullToDelete)) == True:
                            droppedList = droppedList[: -1]
                        else:
                            dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
                else:
                    st.success("All the rows are complete at least for the 60%!")
                st.markdown("---")
                for i in range(0, len(correlationList)):
                    if correlationList[i][0] in dfAutomatic.columns and correlationList[i][1] in dfAutomatic.columns:
                        if correlationSum[correlationList[i][0]] > correlationSum[correlationList[i][1]]:
                            x = 0
                            y = 1
                        else:
                            x = 1
                            y = 0
                        f"save_button_{correlationList[i][0]}_{correlationList[i][1]}"
                        strDropAutomaticCorrLoad = "Dropping column " + correlationList[i][
                            x] + " because of it's high correlation with column " + correlationList[i][y]
                        strDropAutomaticCorrConfirmed = f"Successfully dropped column **{correlationList[i][x]}** because of its high correlation with column {correlationList[i][y]}"
                        strDropCorrRollback = f"Check to rollback the drop of column **{correlationList[i][x]}**"
                        # dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                        droppedList.append(["column", correlationList[i][x]])
                        if st.session_state['once'] == True:
                            with st.spinner(text=strDropAutomaticCorrLoad):
                                time.sleep(0.5)
                        st.success(strDropAutomaticCorrConfirmed)
                        with st.expander("Why I did it?"):
                            st.write(
                                "When two columns has an high correlation between each other, this means that the 2 of them together have almost the same amount of information with respect to have only one of them. ANyway some columns can be useful, for example, to perform aggregate queries. If you think it's the case with this column you should better rollback this action and keep it!")
                            if st.checkbox(strDropCorrRollback, key=NoDupKey) == True:
                                droppedList = droppedList[: -1]
                            else:
                                dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                            NoDupKey = NoDupKey - 1
                        # st.markdown("<p id=page-bottom>You have reached the bottom of this page!!</p>", unsafe_allow_html=True)
                        st.markdown("---")
                for col in dfAutomatic.columns:
                    # k = randint(1,100)
                    if len(pd.unique(dfAutomatic[col])) == 1:
                        strDropAutomaticDistLoad = "Dropping column " + col + " because has the same value for all the rows, that is " + str(
                            dfAutomatic[col][1])
                        strDropAutomaticDistConfirmed = f"Successfully dropped column **{col}** because has the same value for all the rows, that is {dfAutomatic[col][1]}"
                        strDropDistRollback = f"Check to rollback the drop of column **{col}**"
                        droppedList.append(["column", col])
                        if st.session_state['once'] == True:
                            with st.spinner(text=strDropAutomaticDistLoad):
                                time.sleep(0.5)
                        st.success(strDropAutomaticDistConfirmed)
                        with st.expander("Why I did it?"):
                            st.write(
                                "The fact that all the rows of the dataset had the same value for this attribute, doesn't bring any additional information with respect to removing the attribute. A dumb example could be: imagine a table of people with name, surname, date of birth...Does make sense to add a column called 'IsPerson'? No, because the answer would be the same for all the rows, we already know that every row here represent a person.")
                            if st.checkbox(strDropDistRollback, key=100) == True:
                                droppedList = droppedList[: -1]
                            else:
                                dfAutomatic = dfAutomatic.drop(col, axis=1)
                        st.markdown("---")
                for col in dfAutomatic.columns:
                    nullNum = dfAutomatic[col].isna().sum()
                    distinct = dfAutomatic[col].nunique()
                    percentageNull = nullNum / len(st.session_state.my_dataframe.index) * 100
                    if percentageNull > 1:
                        if dfAutomatic[col].dtype == "object":  # automatically fill with the mode
                            x = 0
                        elif dfAutomatic[col].dtype == "float64" or dfAutomatic[
                            col].dtype == "Int64":  # automatically fill with the average
                            x = 1
                        else:
                            x = 2
                            st.error("Unrecognized col. type")
                        if x != 2:
                            strFillAutomaticLoad = "Replacing all the " + str(nullNum) + " (" + "%0.2f" % (
                                percentageNull) + "%) null values of column " + col
                            strFillAutomaticRollback = f"Check to rollback the replacement of all the null values in column **{col}**"
                            originalCol = dfAutomatic[col].copy(deep=False)
                        if x == 0:
                            try:
                                strMode = report["variables"][col]["top"]
                                dfAutomatic[col].fillna(strMode, inplace=True)
                                strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str(
                                    "%0.2f" % (
                                        percentageNull)) + f"%) null values of the column **{col}** with the mode: {strMode}"
                                explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible this attribute, the mode is the value less invasive in terms of filling.  In the null values you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                            except:
                                ()
                        elif x == 1:
                            avgValue = "{:.2f}".format(report["variables"][col]["mean"])
                            dfAutomatic[col].fillna(round(round(float(avgValue))), inplace=True)
                            strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str("%0.2f" % (
                                percentageNull)) + f"%) null values of the column **{col}** with the average value: {round(float(avgValue))}"
                            explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible the average value of this attribute, the mean is one of the best solution for the replacement. In the null values page you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                        if x == 0 or x == 1:
                            droppedList.append(["nullReplaced", col, dfAutomatic[col]])
                            if st.session_state['once'] == True:
                                with st.spinner(text=strFillAutomaticLoad):
                                    time.sleep(0.5)
                            st.success(strFillAutomaticConfirmed)
                            with st.expander("Why I did it?"):
                                st.write(explanationWhy)
                                k = nullNum + distinct
                                if st.checkbox(strFillAutomaticRollback, key=k) == True:
                                    droppedList = droppedList[: -1]
                                else:
                                    if x == 0:
                                        dfAutomatic[col].fillna(dfAutomatic[col].mode(), inplace=True)
                                    elif x == 1:
                                        dfAutomatic[col].fillna(avgValue, inplace=True)
                        length = round(len(dfAutomatic.index) / 10)
                        limit = round(length * 60 / 100)
                        redundancyList = []
                        for col in dfAutomatic.columns:
                            for col1 in dfAutomatic.columns:
                                if col != col1:
                                    dup = 0
                                    for i in range(length):  #reindex?
                                        if i in dfAutomatic.index:
                                            if str(dfAutomatic[col][i]) in str(
                                                    dfAutomatic[col1][i]):  # col1/arg1 ubicazione, col/arg descrizioneVia
                                                dup += 1
                                    if dup > limit:
                                        # st.write(f"The column  **{col1}** cointans the ", "%0.2f" %(percentageDup), "%" + " of the information present in the column " + f"**{col}**")
                                        redundancyList.append([col, col1])
                        intk = 200
                        flag = 0
                        for item in redundancyList:
                            flag = 1
                            strRemoveRedLoad = "Removing the redundancy of information between column " + item[
                                0] + " and " + item[
                                                   1]
                            strRemoveRedConfirmed = f"Successfully removed all the redundancy of information between **{item[0]}** and **{item[1]}**! Now the information is present only in column **{item[0]}**."
                            strRemoveRedRollback = f"Check to restore the information in column **{item[1]}**"
                            if st.session_state['once'] == True:
                                with st.spinner(text=strRemoveRedLoad):
                                    time.sleep(1)
                            st.success(strRemoveRedConfirmed)
                            with st.expander("Why I did it?"):
                                st.write(
                                    "The two columns were partially representing the same instances. So the redundant information was dropped from the most complete column. This because it's usually best practise to do not aggregate too much information within only one column.")
                                if st.checkbox(strRemoveRedRollback, key=intk) == True:
                                    droppedList = droppedList[: -1]
                                else:
                                    for i in range(len(dfAutomatic.index)):
                                        if str(dfAutomatic[item[0]][i]) in str(dfAutomatic[item[1]][i]):
                                            try:
                                                dfAutomatic[item[1]][i] = str(dfAutomatic[item[1]][i]).replace(
                                                    str(dfAutomatic[item[0]][i]), "")
                                                intk += 1
                                            except:
                                                intk += 1

                        st.info("No other actions to be perfomed")
                        st.markdown("---")
                        st.subheader("New dataset real time preview")
                        st.write(dfAutomatic)
                        st.session_state['newdf'] = dfAutomatic.copy()
                        st.warning(
                            "If you see columns with poor information you've the chance to drop them. Remind that you're also applying *permanently* all the changes above.")
                        colSave, colSaveDrop, colIdle = st.columns([1, 1, 8], gap='small')
                        with colSave:
                            button_key1 = str(uuid.uuid4())
                            button_key2 = str(uuid.uuid4())

                            # Now you can use button_key in your st.button call
                            if st.button("Save", key=button_key1, on_click=clean2):
                                ()
                                # st.session_state['y'] = 2
                                # st.session_state['toBeProfiled'] = True
                                # st.experimental_rerun()
                        with colSaveDrop:
                            if st.button("Save and go to Drop", key=button_key2, on_click=clean3):
                                ()
                                # st.session_state['y'] = 3
                                # st.experimental_rerun()
                        st.session_state['once'] = False
                        st.markdown("---")
                        button_key_back_to_homepage1 = str(uuid.uuid4())

                        # Use the unique key in your st.button call
                        if st.button("Back To Homepage", key=button_key_back_to_homepage1):
                            switch_page("homepage")

                        # elif st.session_state['y'] == 3:
                    if st.session_state['y'] == 3:
                        dfAutomatic = st.session_state['newdf']
                        listCol = dfAutomatic.columns
                        listCol = listCol.insert(0, "Don't drop anything")
                        colToDrop = st.multiselect("Select one or more column to drop", listCol, "Don't drop anything")
                        if (len(colToDrop) > 0) and (colToDrop.count("Don't drop anything") == 0):
                            # if st.button("Save"):
                            #    st.session_state['newdf'] = dfAutomatic.copy()
                            #    st.session_state['y'] = 2
                            #    st.session_state['toBeProfiled'] = True
                            #    st.experimental_rerun()
                            # else:
                            for col in colToDrop:
                                dfAutomatic = dfAutomatic.drop(col, axis=1)
                        st.subheader("Real time preview")
                        st.write(dfAutomatic.head(50))
                        col1, col2, col3 = st.columns([1, 1, 8], gap='small')
                        with col1:
                            if st.button("Save"):
                                st.session_state['newdf'] = dfAutomatic.copy()
                                st.session_state['y'] = 2
                                st.session_state['toBeProfiled'] = True
                                st.experimental_rerun()
                        with col2:
                            if st.button("Back"):
                                st.session_state['y'] = 0
                                st.session_state['once'] = True
                                st.experimental_rerun()
                        if st.session_state['y'] == 2:
                            successMessage = st.empty()
                            successString = "Please wait while the dataframe is profiled again with all the applied changes.."
                            st.session_state.my_dataframe = st.session_state['newdf']
                            if st.session_state['toBeProfiled'] == True:
                                successMessage.success(successString)
                                # st.markdown('''(#automatic)''', unsafe_allow_html=True)
                                with st.spinner(" "):
                                    profileAgain(st.session_state.my_dataframe)
                            successMessage.success(
                                "Profiling updated! You can go to 'dataset info' in order to see the report of the new dataset or comeback to the homepage.")
                            st.session_state['toBeProfiled'] = False
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 1, 10], gap='small')
                            with col1:
                                button_key_back_to_homepage2 = str(uuid.uuid4())

                                # Use the unique key in your st.button call
                                if st.button("Back To Homepage", key=button_key_back_to_homepage2):
                                    switch_page("homepage")
                            with col2:
                                if st.button("Dataset Info"):
                                    switch_page("dataset_info")
                        # st.markdown("---")

if scelta_pagina == "Supported Cleaning":


    st.header("Supported Cleaning")
    col1, col2, col3 = st.columns(3)

    with col1:
    #    techniques = {
    #       "Completeness": {
    #          "Imputation - Imputation using functional dependencies": "imputation - imputation usingg functional dependencies",
    #            "Imputation - Mode Imputation": "imputation - mode imputation",
    #            "Imputation - softimpute imputation" : "imputation - softimpute imputation",
    #            "Imputation - Random Imputation" : "imputation - random imputation",
    #            "Imputation - No imputation" : "imputation - no imputation ",
    #            "Imputation - Linear and Logistic Regression Imputation" : "imputation - linear and logistic regression imputation",
    #            "Imputation - Logistic Regression Imputation" : "imputation - logistic regression imputation",
    #            "Imputation - Std imputation" : "imputation - std imputation",
    #            "Imputation - Standard Value Imputation" :"imputation - standard value imputation",
    #            "Imputation - Median Imputation" : "imputation - median imputation",
    #            "Imputation - Mean Imputation" : "imputation - mean imputation",
    #          "Imputation - Mice Imputation" : "imputation - mice imputation"


    #        },
    #        "Accuracy": {
    #            "Normalize Data": "normalize_data",
    #            "Detect and Correct Outliers": "outlier_correction"
    #        },
    #        "Consistency": {
    #            "Other Techniques...": "other_techniques"
    #        }
    #    }


        acc = accuracy_value(st.session_state.my_dataframe)
        com = completeness_value(st.session_state.my_dataframe)
        con = consistency_value
        kbrank = rank_kb(st.session_state.my_dataframe,st.session_state.selected_ml_technique)
        assrank = rank_dim(acc,con,com)
        techniques = average_ranking(kbrank,assrank)
        # Create separate selectboxes for the three sections with the same choices
        tab_section_1 = st.selectbox("Select section (1):", techniques, index=0)
        tab_section_2 = st.selectbox("Select section (2):", techniques, index=1)
        tab_section_3 = st.selectbox("Select section (3):", techniques, index=2)

    with col2:

        output_data = prova_db()
        actions_1 = [item["text"] for item in output_data if item["dimension"] == tab_section_1]
        actions_2 = [item["text"] for item in output_data if item["dimension"] == tab_section_2]
        actions_3 = [item["text"] for item in output_data if item["dimension"] == tab_section_3]

        # Mostra le multiselect per le azioni
        selected_actions_1 = st.multiselect(f"Suggested actions for {tab_section_1}:", actions_1, default=actions_1[4])
        selected_actions_2 = st.multiselect(f"Suggested actions for {tab_section_2}:", actions_2, default="Imputation - No Imputation")
        selected_actions_3 = st.multiselect(f"Suggested actions for {tab_section_3}:", actions_3, default=actions_3[0])

        if "selected_actions" not in st.session_state:
            st.session_state.selected_actions = []

            # Combine selected actions from all three sections
        selected_actions = selected_actions_1 + selected_actions_2 + selected_actions_3

        # Remove deselected items from the pipeline in the correct order
        removed_items = [item for item in st.session_state.selected_actions if item not in selected_actions]
        st.session_state.selected_actions = selected_actions

        # Initialize the data preparation pipeline if not already present
        if "data_preparation_pipeline" not in st.session_state:
            st.session_state.data_preparation_pipeline = []

        # Remove deselected items from the pipeline
        for item in removed_items:
            if item in st.session_state.data_preparation_pipeline:
                st.session_state.data_preparation_pipeline.remove(item)


    with col3:
        # Show the ordered pipeline as text (tag-like)
        st.write("Ordered Data Preparation Pipeline:")
        bestcomp = prova_classifier()
        azione_stringa = bestcomp[0]







            # Aggiungi le azioni selezionate alla pipeline nell'ordine corretto
        for technique in selected_actions:
            if technique in st.session_state["data_preparation_pipeline"]:
                continue  # Evita di aggiungere tecniche duplicate
            st.session_state["data_preparation_pipeline"].append(technique)


            # Visualizza la pipeline nell'ordine in cui le azioni sono state selezionate
        for index, technique_name in enumerate(st.session_state["data_preparation_pipeline"]):
            st.write(f"{index + 1}. {technique_name}")

outlier_range = st.session_state.intervals
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
num_actions = len(st.session_state["data_preparation_pipeline"])



if st.session_state.current_index < num_actions:

    current_action = st.session_state["data_preparation_pipeline"][st.session_state.current_index]
    st.write(f"Current action: {current_action}")

    # Esegui l'azione corrente sulla copia del dataframe
    temp_dataframe = st.session_state.my_dataframe.copy()
    temp_dataframe = save_and_apply(current_action, temp_dataframe, outlier_range)
    st.write(temp_dataframe)

    # Bottone per confermare l'azione corrente
    if st.button(f"Confirm Action "):
        st.session_state.my_dataframe = temp_dataframe
        st.session_state.current_index = min(st.session_state.current_index + 1, num_actions)
        if st.button("save confirm"):
            st.write("Action confirmed successfully")

        # Bottone per fare rollback all'azione precedente
    if st.button(
            f"Rollback to Previous Action") and st.session_state.current_index > 0:
        st.session_state.current_index = max(st.session_state.current_index - 1, 0)
        if st.button("save rollback"):
            st.write("Rolled back to previous action")

    st.markdown("---")

st.write("---")

if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Duplicate")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Functional_Dependencies")




