# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

# Flask modules
#from msvcrt import kbhit
from flask   import Flask, session, render_template, request, abort, redirect, url_for, flash, send_from_directory,send_file, jsonify, Markup
from jinja2  import TemplateNotFound
from PIL import Image
import os
import tempfile
import lime
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import lime
import lime.lime_tabular
import pandas as pd
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from joblib import load
import lime.lime_tabular
from sklearn.pipeline import make_pipeline


from ydata_profiling import ProfileReport
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import apps.scripts.kb_test as kb
import apps.scripts.allThePlots as myPlots
import apps.scripts.quality_dimensions_and_ranking_calc as dims_and_rank
import apps.scripts.improve_quality_laura as improve
import apps.scripts.data_imputation_tecniques as imputes
from apps.scripts.preprocessing import preprocess_input_df
from flask_session import Session
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openai
from neo4j import GraphDatabase
from joblib import dump, load
from pprint import pprint
'''
def descriptionDatset():
    api_key = "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"
    dataset = pd.read_csv('apps/datasets/beers.csv')
    # Estrai automaticamente le caratteristiche dal dataset
    features = dataset.describe()
    # Create a question for ChatGPT
    domanda = f" Describe in detail  the attributes of each column in this dataset to introduce it to a non-expert user :\n{features.to_string()}."
    openai.api_key = api_key  # Imposta la chiave API
    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": domanda},
        ],
    )

    # Estrai la risposta senza alcuna formattazione
    formatted_response = risposta['choices'][0]['message']['content'].strip()

    # Formatta la risposta come elenco puntato in HTML
    formatted_html = f"<div>{'<br>'.join([f'<p>{line}</p>' for line in formatted_response.splitlines()])}</div>"
    # Restituisci la risposta formattata in HTML
    return formatted_html
'''

# App modules
from apps import app

@app.template_filter('json_round')
def json_round_filter(json_obj):
    def _json_round(obj):
        if isinstance(obj, float):
            return round(obj, 2)
        elif isinstance(obj, dict):
            return {_json_round(key): _json_round(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_json_round(elem) for elem in obj]
        else:
            return obj

    return _json_round(json_obj)



HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_DATA = {
    'iris': os.path.join(HERE, 'datasets/iris.csv'),
    'beers': os.path.join(HERE, 'datasets/beers.csv'),
}


DF_NAME = 'datasets/data.csv'
DF_NAME_TEMP = 'datasets/data_temp.csv'

app.secret_key = 'my_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

dataframes = [] #global variable
column_list = [0]

information_to_print = "Click here to get information about the selected techniques!"



def get_columns(columns):
    if isinstance(columns, str):        
        columns = columns.replace("[", "")
        columns = columns.replace("]", "")
        columns = columns.replace("'", "")
        columns = columns.replace(" ", "")
        columns = columns.split(",")
    return columns

def save_data(df):
    # save the cleaned dataset as csv
    df.to_csv(os.path.join(HERE, DF_NAME))

def save_data_temp(df):
    # save the cleaned dataset as csv
    df.to_csv(os.path.join(HERE, DF_NAME_TEMP))

def get_data(name, dirname):
    upload_path = os.path.join(tempfile.gettempdir(), dirname)
    data_path = os.path.join(upload_path, name + '.csv')
    if not os.path.exists(data_path):
        abort(404)

    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError:
        flash('Bad CSV file – could not parse', 'warning')
        return redirect(url_for('home'))

    (df, columns) = preprocess_input_df(df)

    return df, columns

 # Get the last DataFrame
def last_dataframe():
    global dataframes
    df = dataframes[-1]
    return df
    
   
    # df_list = session.get('df_list', [])
    # # last_df = df_list[-1] if len(df_list) > 0 else None

    # last_element = df_list[-1]

    # # convert the last element to a DataFrame object
    # df = pd.DataFrame(last_element)
    
    # Render a template that displays the DataFrame
   

# Function to append dataframe to the global list
def append_dataframe(df):
    global dataframes
    dataframes.append(df)
    


#delete json report once the upload or new upload is pressed
# @app.route('/delete_file', methods=['POST'])
# def delete_file():
       
#     # Create a relative file path using os.path.join()
#     file_path = os.path.join(HERE, 'report/profile.json')

#     # Delete the file
#     os.remove(file_path)
    
#     # Return a JSON response with a success message
#     response = jsonify({'message': 'File deleted successfully'})
#     return response

def descriptionDatset():
    api_key = "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"
    dataset = pd.read_csv('apps/datasets/beers.csv')
    # Estrai automaticamente le caratteristiche dal dataset
    features = dataset.describe()
    # Create a question for ChatGPT
    domanda = f" Describe in detail  the attributes of each column in this dataset to introduce it to a non-expert user :\n{features.to_string()}."
    openai.api_key = api_key  # Imposta la chiave API
    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": domanda},
        ],
    )

    # Estrai la risposta senza alcuna formattazione
    formatted_response = risposta['choices'][0]['message']['content'].strip()

    # Formatta la risposta come elenco puntato in HTML
    # formatted_html = f"<div>{'<br>'.join([f'<p>{line}</p>' for line in formatted_response.splitlines()])}</div>"
    formatted_html = f"""<div>{''.join([f'<p style="margin-bottom: 3px;">{line}</p>' for line in formatted_response.splitlines()])}</div>"""
    # Restituisci la risposta formattata in HTML
    return formatted_html


def get_schematic_description():
    api_key = "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"
    dataset = pd.read_csv('apps/datasets/beers.csv')
    # Estrai automaticamente le caratteristiche dal dataset
    features = dataset.describe()
    # Create a question for ChatGPT
    domanda = f"Provide a very very short and summarized description of  dataset in a schematic  format :\n{features.to_string()}."
    openai.api_key = api_key  # Imposta la chiave API
    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": domanda},
        ],
    )

    # Estrai la risposta senza alcuna formattazione
    formatted_response = risposta['choices'][0]['message']['content'].strip()

    # Formatta la risposta come elenco puntato in HTML
    # formatted_html = f"<div>{'<br>'.join([f'<p>{line}</p>' for line in formatted_response.splitlines()])}</div>"
    formatted_html = f"""<div>{''.join([f'<p style="margin-bottom: 3px;">{line}</p>' for line in formatted_response.splitlines()])}</div>"""
    # Restituisci la risposta formattata in HTML
    return formatted_html





def detectMissingValues():

    api_key = "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"
    dataset = pd.read_csv('apps/datasets/beers.csv')

    missing_values = dataset.isnull().sum()

    # print("Valori Mancanti per ciascuna colonna:")
    # print(missing_values)

    domanda = f"Explain to a non-expert user what are the most relevant pieces of information they should know given this missing data in the dataset.:\n{missing_values.to_string()}"

    openai.api_key = api_key  # Imposta la chiave API
    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": domanda},
        ],
    )

    return risposta['choices'][0]['message']['content'].strip()


def detectCorrelation():
    api_key = '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'
    df = pd.read_csv('apps/datasets/beers.csv')

    correlation_matrix = df.corr(numeric_only=True)
    correlation_text = correlation_matrix.to_string()
    question = f"Explain in a very detailed way the most relevant insights from the correlation matrix below to a non expert user:\n{correlation_text}"

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )

    return response['choices'][0]['message']['content'].strip()

'''
def get_summary_correlation():
    api_key = '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'
    df = pd.read_csv('apps/datasets/beers.csv')

    correlation_matrix = df.corr(numeric_only=True)
    correlation_text = correlation_matrix.to_string()
    question = f"Provide a very  short Explanation of  the  correlation matrix  to a non expert user:\n{correlation_text}"

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )

    return response['choices'][0]['message']['content'].strip()

'''

def detectOutliers():

    api_key = "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"

    dataset = pd.read_csv('apps/datasets/beers.csv')

    Q1 = dataset.quantile(0.25, numeric_only=True)
    Q3 = dataset.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1

    soglia = 1.5

    # Calcola gli outlier per valori inferiori
    outliers_lower = dataset.lt(Q1 - soglia * IQR)

    # Calcola gli outlier per valori superiori
    outliers_upper = dataset.gt(Q3 + soglia * IQR)

    # Combina gli outlier per valori inferiori e superiori
    outliers = outliers_lower | outliers_upper

    # print("Outlier per ciascuna colonna:")
    # print(outliers.sum())

    domanda = f"Explain to a non-expert user what are the most relevant pieces of information they should know given the outliers in the dataset:\n{outliers.sum().to_string()}"

    openai.api_key = api_key

    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": domanda},
        ],
    )

    return risposta['choices'][0]['message']['content'].strip()
# App main route + generic routing
@app.route('/', methods=['GET'])
def home():
    return render_template('home/index.html')

@app.route('/upload.html', methods=['GET'])
def upload():
    return render_template('home/upload.html')

@app.route('/audit', methods=['POST'])
def upload_file():
    session.clear()
    referer = request.headers.get('referer')
    redirect_url = referer or url_for('home/index.html')

    file_ = request.files.get('file')

    if not file_ or not file_.filename:
        flash('Please select a file', 'warning')
        return redirect(redirect_url)

    (name, ext) = os.path.splitext(file_.filename)
    if not ext.lower() == '.csv':
        flash('Bad file type – CSV required', 'warning')
        return redirect(redirect_url)

    dirpath = tempfile.mkdtemp(prefix='')
    filename = secure_filename(file_.filename)
    file_.save(os.path.join(dirpath, filename))
    
    if request.method == 'POST':
        (df, columns) = get_data(name, dirpath)
        myPlots.table_df(df)
        return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))



#the user can also select one of the existing datasets
@app.route('/audit/<name>/', methods=['GET'])
def upload_sample(name):
    if name not in SAMPLE_DATA:
        abort(404)

    source_path = SAMPLE_DATA[name]
    filename = os.path.basename(source_path)
    (name, _ext) = os.path.splitext(filename)
    dirpath = tempfile.mkdtemp(prefix='')
    dest_path = os.path.join(dirpath, filename)
    os.symlink(source_path, dest_path) 
    
    return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))




@app.route('/audit/<name>/<dirname>/', methods=['GET', 'POST'])
def audit_file(name, dirname):
    (df, columns) = get_data(name, dirname)
    session['name'] = name
    session['dirname'] = dirname

    selected_attributes = []
       
    algorithm = request.form.get("algorithm")
    session['algorithm'] = algorithm
    risp_description = descriptionDatset()
    risp_description_s=get_schematic_description()

    for col in columns:
        if col == request.form.get(col):
            selected_attributes.append(col)

    #  store selected_attributes in session
    session['selected_attributes'] = selected_attributes

    # global column_list
    # column_list = get_columns(selected_attributes)
    
    wants_support=request.form.get('req_support')

    # if request.form.get('Support'):
    #     support = request.form.get('Support')
    # else: support = 80
    support=80
    session['support'] = support
    
    # if request.form.get('Confidence'):
    #     confidence = request.form.get('Confidence')
    # else: confidence = 90
    confidence = 90
    session['confidence'] = confidence

    
    
    df = df[:][selected_attributes]
    dataFrame= save_data(df)
    append_dataframe(df)
    
    # Append the modified DataFrame to the session list
    # df_copy = df.copy()
    # session['df_list'] = []
    # session['df_list'].append(df_copy.to_dict(orient='records'))
    if str(request.form.get("submit")) == "Upload new dataset":
        session.clear()
        return redirect(url_for('upload'))
    
    if str(request.form.get("submit")) == "Profiling":
        profile = ProfileReport(df)
        
        #code for multiple profiles
        # i=len(dataframes)
        # profile.to_file(f'apps/report/profile_{i}.json')

        # code for single profile
        profile.to_file('apps/report/profile.json')

        

        return redirect(url_for("data_profiling",
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                        wants_support=wants_support,
                    
                        support=support,
                        confidence=confidence
                            )
                        )

    else:

        return render_template("home/audit.html",
                                name=name,
                                dirname=dirname,
                                columns=columns,
                                risp_description = descriptionDatset(),
                                risp_description_s=get_schematic_description()
                                )
    



@app.route('/submit_outliers', methods=['POST'])
def submit_outliers():
    form_data = request.get_json() # read the form data from the AJAX request
    # do some processing with the form data (e.g., detect outliers)
    response_data = {'message': 'Outliers detected.'} # create a response message
    print(form_data)
    min_values = []
    max_values = []
    min_values = [float(v) for k, v in form_data.items() if k.startswith('min_')]
    max_values = [float(v) for k, v in form_data.items() if k.startswith('max_')]
   
  # Process the form data here
  # Return a response
    session['min_values'] = min_values
    session['max_values'] = max_values
    
    print(session['min_values'])
    print(session['max_values'])
    return jsonify(response_data) # send the response back to the JavaScript frontend


def get_techniques(ml_algorithm):

    # kb interaction

    URI = "neo4j://localhost"
    AUTH = ("neo4j", "ciaociao")

    driver = GraphDatabase.driver(URI, auth=AUTH)

    driver.verify_connectivity()
    print("verify connectivity tutto ok\n")

    dimensions = ["Uniqueness", "Completeness", "Accuracy"]

    techniques = [
        # {"id": "remove_duplicates", "text": "Remove duplicates", "dimension":"UNIQUENESS" },

        # {"id": "impute_standard", "text": "Imputation (0/Missing)", "dimension":"COMPLETENESS"},
        # da rimettere drop rows !
        # {"id": "drop_rows", "text": "Drop rows with missing values", "dimension":"COMPLETENESS"},
        # {"id": "impute_mean",  "text": "Imputation (mean/mode)", "dimension":"COMPLETENESS"},
        # {"id": "impute_std",  "text": "Imputation (standard deviation/mode)", "dimension":"COMPLETENESS"},
        # {"id": "impute_mode",  "text": "Imputation (mode)", "dimension":"COMPLETENESS"},
        # {"id": "impute_median",  "text": "Imputation (median/mode)", "dimension":"COMPLETENESS"},
        # {"id": "impute_knn", "text": "Imputation using KNN", "dimension":"COMPLETENESS"},
        # {"id": "impute_mice",  "text": "Imputation using Mice", "dimension":"COMPLETENESS"},
        
        # {"id": "outlier_correction", "text": "Outlier correction", "dimension":"ACCURACY"},
        # {"id": "oc_impute_standard",  "text": "Outlier correction with imputation (0/Missing)", "dimension":"ACCURACY"},
        # {"id": "oc_drop_rows",  "text": "Outlier correction with drop rows", "dimension":"ACCURACY"},
        # {"id": "oc_impute_mean", "text": "Outlier correction with imputation (mean/mode)", "dimension":"ACCURACY"},
        # {"id": "oc_impute_std", "text": "Outlier correction with imputation (standard deviation/mode)", "dimension":"ACCURACY"},
        # {"id": "oc_impute_mode", "text": "Outlier correction with imputation (mode)", "dimension":"ACCURACY"},
        # {"id": "oc_impute_median", "text": "Outlier correction with imputation (median/mode)", "dimension":"ACCURACY"},
        # {"id": "oc_impute_knn", "text": "Outlier correction with imputation (KNN)", "dimension":"ACCURACY"},
        # {"id": "oc_impute_mice", "text": "Outlier correction with imputation (Mice)", "dimension":"ACCURACY"}
    ]

    for dimension in dimensions:

        # Get the techniques that improve that dimension
        records, summary, keys = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[a:AFFECTS]->(d:DQ_DIMENSION) \
            WHERE a.influence_type = $influence_type and d.name = $dimension_name \
            RETURN n.name AS name",
            influence_type="Improvement",
            dimension_name=dimension,
            database_="neo4j",
        )
        for tech in records:
            # print(tech["name"])
            # Here I query the methods for that technique
            records_m, summary_m, keys_m = driver.execute_query(
                "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
                WHERE n.name = $technique_name \
                RETURN m.name AS name",
                technique_name=tech["name"],
                database_="neo4j",
            )
            for meth in records_m:
                # print(meth["name"])
                # print({"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": dimension.upper()})
                techniques.append({"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": dimension.upper()})

    # reformat the ml algorithms' names to query the kb
    if ml_algorithm == "DT": ml_algorithm = "Decision Tree"
    if ml_algorithm == "NB": ml_algorithm = "Naive Bayes"

    records, summary, keys = driver.execute_query(
        "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:BENEFITS_FROM]-(ml:ML_APPLICATION) \
         WHERE ml.application_method = $ml_algorithm \
         RETURN DISTINCT n.name AS name",
        ml_algorithm=ml_algorithm,
        database_="neo4j"
    )
    for tech in records:
        # print(tech["name"])
        # Here I query the methods for that technique
        records_m, summary_m, keys_m = driver.execute_query(
            "MATCH (n:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
            WHERE n.name = $technique_name \
            RETURN m.name AS name",
            technique_name=tech["name"],
            database_="neo4j",
        )
        for meth in records_m:
            # print(meth["name"])
            techniques.append({"id": meth["name"], "text": tech["name"] + " - " + meth["name"], "dimension": "ML_ORIENTED_ACTIONS"})

    # print(best_imputation_method())

    driver.close()
    return techniques

# {"id": "oc_drop_cols", "name": "", "text": "Outlier correction with drop columns", "dimension":"ACCURACY"},
# {"id": "drop_cols", "name": "", "text": "Drop columns with missing values", "dimension":"COMPLETENESS"},
def best_imputation_method():

    trained_model = load('trained_classifier.joblib') #viene caricato un modello già allenato che mi calcola il best method

    dataset = pd.read_csv("dataset_classifier_features.csv")

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

    scaler = load('trained_scaler.joblib')
    # scaler = StandardScaler()
    # scaler = RobustScaler()

    dataset = scaler.transform(dataset)

    dataset = pd.DataFrame(dataset, columns=feature_cols)

    best_method_predicted = trained_model.predict(dataset)

    print("best imputation method " + best_method_predicted[0])
    print("KB name: " + imputation_name_conversion(best_method_predicted[0]) + "\n")

    return imputation_name_conversion(best_method_predicted[0])


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
import lime.lime_tabular
import numpy as np

def best_imputation_method1():
    import pandas as pd
    import numpy as np
    from joblib import load
    import lime.lime_tabular
    from joblib import load
    import matplotlib.pyplot as plt

    X_train = pd.read_csv("KB_whole_datasets_no_fill_preprocessing_mode.csv")
    ml_algorithm_column = X_train['ML_ALGORITHM']
    # Estrai la colonna BEST_METHOD
    y_train = X_train["BEST_METHOD"]

    X_train = X_train.drop(["name", "BEST_METHOD", "ML_ALGORITHM"], axis=1)

    # dataset = pd.DataFrame(dataset, columns=feature_cols)
    # Carica il dataset da spiegare
    dataset = pd.read_csv("dataset_classifier_features.csv")
    dataset = pd.get_dummies(dataset, columns=['ML_ALGORITHM'])
    # Modifico datset cosi ha la stessa struttura delle fature_cols
    ml_columns = ["ML_ALGORITHM_dt", "ML_ALGORITHM_lr", "ML_ALGORITHM_knn", "ML_ALGORITHM_nb"]

    # inserisco quelle 4 colonne con valore 0
    for c in ml_columns:
        dataset[c] = 0
    feature_cols = dataset.columns
    [
        'n_tuples', 'n_attributes', 'p_num_var', 'p_cat_var', 'p_duplicates',
        'total_size', 'p_avg_distinct', 'p_max_distinct', 'p_min_distinct',
        'avg_density', 'max_density', 'min_density', 'avg_entropy', 'max_entropy',
        'min_entropy', 'p_correlated_features', 'max_pearson', 'min_pearson',
        'ML_ALGORITHM_dt', 'ML_ALGORITHM_lr', 'ML_ALGORITHM_nb', 'ML_ALGORITHM_knn'
    ]

    # Aggiungi le colonne mancanti a X_train con valori estratti dalla colonna ML_ALGHORITM presente nell originario x_train
    missing_columns = set(dataset.columns) - set(X_train.columns)

    # for column in missing_columns:
    for column in ml_columns:
        # Controlla se la colonna è una colonna dummy di ml_algorithm
        if column.startswith("ML_ALGORITHM_"):
            ml_algorithm = column.replace("ML_ALGORITHM_", "")
            # Imposta il valore sulla colonna corrispondente di ml_algorithm
            X_train["ML_ALGORITHM_" + ml_algorithm] = (ml_algorithm_column == ml_algorithm).astype(int)
        else:
            X_train[column] = 0

    # Trasforma alcune colonne in modo specifico, per non avere valori negativi
    X_train['min_pearson'] = X_train['min_pearson'].apply(lambda x: x if x >= 0 else abs(x))
    X_train = X_train.fillna(X_train.mean())
    dataset['min_pearson'] = dataset['min_pearson'].apply(lambda x: x if x >= 0 else abs(x))

    # Aggiungi nomi di feature a X_train
    dataset = dataset.drop(columns=["name"])
    # Carica il modello allenato
    trained_model = load('trained_classifier.joblib')
    # Carica il modello di scalatura
    scaler = load('trained_scaler.joblib')

    # Esempio di modifica dei parametri di Lime
    exp = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        mode='classification',
        training_labels=y_train,
        feature_names=feature_cols,
        class_names=y_train.unique().tolist(), )

    feature_cols = dataset.columns

    def custom_return_prob(instances):
        data = instances.copy()
        idxnumcols = [X_train.columns.get_loc(c) for c in feature_cols]
        data[:, idxnumcols] = scaler.transform(instances[:, idxnumcols])
        predicted_probabilities = trained_model.predict_proba(data)
        return predicted_probabilities

    def softmax(x, axis=None):  # per avere la somma delle probabilità uguale a 1
        if axis is None:
            axis = 1
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / exp_x.sum(axis=axis, keepdims=True)

    # per odinare alla stessa maniera le etichette del dataset e quelle prese da lime
    lime_class_order = exp.class_names
    model_classes = np.array(trained_model.classes_)

    # Creare un mapping tra le classi del modello e le classi di Lime
    class_mapping = {lime_class: model_class for lime_class, model_class in zip(lime_class_order, model_classes)}

    # Esegui il modello su un campione (sample)
    sample_idx = 0  # Sostituisci con l'indice della riga desiderata da "dataset_classifier_features.csv"
    sample = dataset.iloc[sample_idx].to_numpy().reshape(1, -1)

    exp.class_names = trained_model.classes_

    # Utilizza la funzione custom_return_prob
    explanation = exp.explain_instance(sample[0], custom_return_prob, top_labels=9)

    # Estrai la spiegazione solo per la label desiderata
    explanation_list = explanation.as_list()

    # Trasforma l'oggetto explanation filtrato in una stringa
    explanation_text = ";\n".join([f"{feat}: {score:.2f}" for feat, score in explanation_list])
    # Formula la domanda a ChatGPT includendo la spiegazione
    domanda = f"These are the results of a computation performed with lime.explanability: {explanation_text}. Provide a comprehensive explanation of each feature in a way that is understandable even for non-experts.  Explain also which of that features  have had the greatest influence on the choice of the best method."

    # Inizializza la sessione di OpenAI
    api_key = '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'
    openai.api_key = api_key

    # Imposta il numero massimo di token per la risposta (ad esempio, 100 o più)
    max_tokens = 2000

    # Inserisci la domanda a ChatGPT online
    risposta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa il modello GPT-3.5 Turbo
        messages=[
            {"role": "user", "content": domanda},
        ],
        max_tokens=max_tokens  # Imposta la lunghezza massima desiderata per la risposta
    )

    # Estrai la risposta senza alcuna formattazione
    formatted_response = risposta['choices'][0]['message']['content'].strip()

    # Formatta la risposta come elenco puntato in HTML
    # formatted_html = f"<div>{'<br>'.join([f'<p>{line}</p>' for line in formatted_response.splitlines()])}</div>"
    formatted_html = f"""<div>{''.join([f'<p style="margin-bottom: 3px;">{line}</p>' for line in formatted_response.splitlines()])}</div>"""
    # Restituisci la risposta formattata in HTML
    return formatted_html

@app.route("/apply/<name>/<dirname>/<algorithm>/<support>/<confidence>/<wants_support>", methods=["POST"])
def save_and_apply(name,dirname,algorithm,support,confidence,wants_support):
    sorted_list = request.get_json()
    # do something with the sorted list
    print(sorted_list)
    df = last_dataframe()

    # global column_list
    # cols= column_list
    cols = session.get('selected_attributes', [])
    min_values = session.get('min_values', [])
    max_values = session.get('max_values', [])
    range_min = []
    range_max = []
    index = 0
    #create range_min and range_max to insert zeros as ranges for the categorical variables(to make outlier correction work)
    for col in cols:
        if (df[col].dtype != "object"):
            range_min.append(min_values[index])
            range_max.append(max_values[index])
            index = index + 1
        else:
            range_min.append(0)
            range_max.append(0)


    outlier_range = [list(x) for x in zip(range_min, range_max)]
    print(outlier_range)

    for tech in sorted_list:
        

        if tech == "remove_duplicates" or tech == "Remove Identical Duplicates" or tech == "Blocking" or tech == "Sorted Neighborhood":
            df = improve.remove_duplicates(df)
            print("Success")
            

        elif tech == "impute_standard" or tech == "Standard Value Imputation":
            # df = improve.imputing_missing_values(df)
            imputator = imputes.impute_standard()
            df = imputator.fit(df)
            print("Success")
            

        elif tech == "drop_rows":
            impute = imputes.drop()
            df = impute.fit_rows(df)
            print("Success")
            

        elif tech == "impute_mean" or tech == "Mean Imputation":
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)
            print("Success")
            

        elif tech == "impute_std" or tech == "Std Imputation":
            impute = imputes.impute_std()
            df = impute.fit_mode(df)
            print("Success")
            

        elif tech == "impute_mode" or tech == "Mode Imputation":
            impute = imputes.impute_mode()
            df = impute.fit(df)
            print("Success")
            

        elif tech == "impute_median" or tech == "Median Imputation":
            impute = imputes.impute_median()
            df = impute.fit_mode(df)
            print("Success")


        elif tech == "impute_knn" or tech == "KNN Imputation":
            impute = imputes.impute_knn()
            df = impute.fit_cat(df)
            # df=df
            print("Success")

            

        elif tech == "impute_mice" or tech == "Mice Imputation":
            impute = imputes.impute_mice()
            df = impute.fit_cat(df)
            # df = df
            print("Success")


        elif tech == "Random Imputation":
            impute = imputes.impute_random()
            df = impute.fit(df)
            # df = df
            print("Success")


        elif tech == "No Impute":
            impute = imputes.no_impute()
            df = impute.fit(df)
            # df = df
            print("Success")


        elif tech == "Linear and Logistic Imputation" or tech == "Linear Regression Imputation" or tech == "Logistic Regression Imputation":
            impute = imputes.impute_linear_and_logistic()
            df = impute.fit(df, df.columns)
            # df = df
            print("Success")


        elif tech == "outlier_correction":
            df = improve.outlier_correction(df, outlier_range)
            print("Success")
            
            
        elif tech == "oc_impute_standard" or tech == "Outliers Standard Value Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_standard()
            df = impute.fit(df)
            print("Success")
            

        elif tech == "oc_drop_rows" or tech == "Drop Outliers' Rows":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.drop()
            df = impute.fit_rows(df)
            print("Success")
            

        elif tech == "oc_impute_mean" or tech == "Outliers Mean Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mean()
            df = impute.fit_mode(df)
            print("Success")
            

        elif tech == "oc_impute_std":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_std()
            df = impute.fit_mode(df)
            print("Success")
            

        elif tech == "oc_impute_mode" or tech == "Outliers Mode Imputation":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_mode()
            df = impute.fit(df)
            print("Success")
            

        elif tech == "oc_impute_median":
            df = improve.outlier_correction(df, outlier_range)
            impute = imputes.impute_median()
            df = impute.fit_mode(df)
            print("Success")
            

        elif tech == "oc_impute_knn" or tech == "Outliers KNN Imputation":
            df = improve.outlier_correction(df, outlier_range)
            # impute = imputes.impute_knn()
            # df = impute.fit(df)
            
            print("Success")
            

        elif tech == "oc_impute_mice":
            df = improve.outlier_correction(df, outlier_range)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            print("Success")

        elif tech == "z-score":
            df = improve.z_score_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            print("Success")

        elif tech == "Min-Max":
            df = improve.min_max_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            print("Success")

        elif tech == "Robust Scaling":
            df = improve.robust_scaler_normalization(df)
            # impute = imputes.impute_mice()
            # df = impute.fit(df)
            print("Success")

    global dataframes
    dataframes.append(df)
    save_data(df) #saves the dataframe in file data.csv
    
    # return print("Changes applied successfully")
    print("Changes applied successfully")

    return redirect(url_for("data_profiling",
                            name=name,
                            dirname=dirname,
                            algorithm=algorithm,
                            wants_support=wants_support,

                            support=support,
                            confidence=confidence
                            )
                    )
    
    





@app.route('/dataprofiling/<name>/<dirname>/<algorithm>/<support>/<confidence>/<wants_support>', methods=['GET', 'POST'])
def data_profiling(name, dirname, algorithm, support, confidence, wants_support):
    # time.sleep(5)
    # access selected_attributes from session
    col_list = session.get('selected_attributes', '0')
    columns=col_list
    
    # access dataframe from global
    # df = last_dataframe()
    global dataframes
    df = dataframes[-1]
    
    global column_list
    if column_list[0]==0:
        columns = get_columns(col_list)
    else:
        columns = column_list
        
    # df = pd.read_csv(os.path.join(HERE, DF_NAME))
    
    df = df[:][columns]
    columns_names= list(df.columns)


    # code to generate a profile at each modification of the dataset
    # i = len(dataframes)
    # profile_path = (f'apps/report/profile_{i}.json')
    # if os.path.isfile(profile_path):
    #     with open(f'apps/report/profile_{i}.json', "r") as f:
    #         json_str = f.read()
    # else:
    #     profile = ProfileReport(df)
    #     profile.to_file(f'apps/report/profile_{i}.json')
    #     with open(f'apps/report/profile_{i}.json', "r") as f:
    #         json_str = f.read()

    
    # code for a single profile
    with open(f'apps/report/profile.json', "r") as f:
        json_str = f.read()
    # Parse the JSON string into a dictionary object
    profile = json.loads(json_str)
    typeList =[]
    typeNUMlist =[]
    typeCATlist =[]
    
    for var in profile['variables'].values():
        typeList.append(var['type'])

    for i in range(len(typeList)):
        if typeList[i]=="Numeric":
            typeNUMlist.append(columns_names[i])
        else: typeCATlist.append(columns_names[i])
    

    # typeNUMlist = df.select_dtypes(include=['int64','float64']).columns
    minValueList = []
    maxValueList = []
    medianList = []
    for var in columns:
        if var in typeNUMlist:
            minValueList.append(df[var].min())
            maxValueList.append(df[var].max()) 
            medianList.append(round(df[var].median(),2))

    #plot generation
    outliers_html_list = myPlots.boxPlot(df, typeNUMlist)  #outlier plots
    myPlots.heatmap(df) #heatmap of the correlation
    distr_html_list = myPlots.distributionPlot(df,typeNUMlist)  #distribution plots(distr_html_list takes the list of html addresses where the plots are saved)
    distrCAT_html_list = myPlots.distributionCategorical(df,typeCATlist)
    #treeMap_html_list= myPlots.treePlot(df, typeCATlist)
    myPlots.missing_data(df, profile) #missigno plots
    myPlots.table_df(df)

    min_values = []
    max_values = []

    if 'min_values' not in session:
        for i in range(len(typeNUMlist)):
            min_values.append(minValueList[i])
        session['min_values'] = min_values
        
    if 'max_values' not in session:
        for i in range(len(typeNUMlist)):
            max_values.append(maxValueList[i])
        session['max_values'] = max_values
        
    min_values = session.get('min_values', [])
    max_values = session.get('max_values', [])
    print(min_values)
    print(max_values)
    
    #calculate dimensions
    accuracy=dims_and_rank.accuracy_value(df, profile, columns, typeNUMlist, min_values, max_values)
    uniqueness=dims_and_rank.uniqueness_value(profile)
    completeness=dims_and_rank.completeness_value(profile, columns)
    
    #ranking of the dimensions
    ranking_dim = dims_and_rank.rank_dim(accuracy, uniqueness, completeness)
   
    #ranking based on the characteristics of the knowledge base
    ranking_kb = dims_and_rank.rank_kb(df, algorithm)

    average_rank = dims_and_rank.average_ranking(ranking_kb, ranking_dim)

    if len(average_rank) <= 3:
        average_rank.append("ML_ORIENTED_ACTIONS")
    else:
        average_rank[3] = "ML_ORIENTED_ACTIONS"
    print("final ranking " + str(average_rank))

    techniques=get_techniques(algorithm)
    risp_missing_val = detectMissingValues()
    risp_correlation = detectCorrelation()
    risp_outliers = detectOutliers()
    risp_description = descriptionDatset()
    risp_description_s = get_schematic_description()
    alertMessage = "[HIGH_CARDINALITY] alert on column name and [HIGH_CARDINALITY] alert on column style: These alerts indicate that the columns 'name' and 'style' contain a large number of different categories. This can complicate the analysis and may require special handling to make sense of the data.\n\n[HIGH_CORRELATION] alert on column abv, [HIGH_CORRELATION] alert on column ibu, [HIGH_CORRELATION] alert on column id, [HIGH_CORRELATION] alert on column brewery_id, [HIGH_CORRELATION] alert on column style: These alerts suggest that some columns in the dataset are closely related to each other. High correlation can affect the accuracy of statistical analyses or models.\n\n[MISSING] alert on column abv and [MISSING] alert on column ibu: These alerts indicate that there are missing values in the 'abv' and 'ibu' columns. Dealing with missing data is crucial, as it can impact the reliability of your analysis.\n\n[UNIFORM] alert on column Unnamed: 0 and [UNIFORM] alert on column name: These alerts mean that some columns contain data that is very similar or shows minimal variation. This uniformity can make it challenging to identify important patterns or differences in the data.\n\n[UNIQUE] alert on column Unnamed: 0 and [UNIQUE] alert on column id: These alerts signify that certain columns contain unique values, which might be crucial for identifying individual items in the dataset"
    #summ_corr=get_summary_correlation()
    exp1="The process of calculating the ranking of quality dimensions involves two initial rankings: RAS and RAP. These rankings are used to determine the order of quality dimensions so that dimensions with lower scores are listed first, and those with higher scores are listed subsequently.\n\nRAS (Ranking from Assessment): This ranking represents the user's evaluation of the dataset's quality. Quality dimensions are listed based on their importance, from least important to most important, according to the user's opinion. For example, if the user considers accuracy more important than completeness, accuracy will be listed first in the ranking.\n\nRAP (Ranking from Application): This ranking is provided by the knowledge base and reflects the importance of quality dimensions for the specific algorithm or application selected. Dimensions are ranked based on their impact on application performance. For example, if a dimension significantly influences algorithm performance, it will be listed first in the ranking.\n\nThe process of merging the two rankings, RAS and RAP, involves assigning different weights to each ranking based on how important the user's opinion is compared to application requirements. For example, if the user values personal preference more, the weight of RAS will be greater. If the application is more influential, the weight of RAP will be higher.\n\nFinally, the system combines the two weighted rankings into a single unified ranking that takes into account both user preferences and application requirements. This final ranking orders the quality dimensions based on this combination of factors, with the most important dimensions for both the user and the application listed at the beginning."
    exp2=best_imputation_method1()
    exp2_summary="The min_pearson feature, with a value of -0.12, stands out as the most influential among the analyzed features in determining the best method. This signifies a weak negative relationship between variables in the dataset. The identification of such a relationship likely guided the choice of a machine learning algorithm capable of effectively handling such dependencies."
    if str(request.form.get("submit")) == "Upload new dataset":
        session.clear()
        return redirect(url_for('upload'))
    
    if str(request.form.get("submit")) == "Modify choices":
        session.clear()
        (df, columns) = get_data(name, os.path.basename(dirname))
        myPlots.table_df(df)

        return redirect(url_for('audit_file', dirname=os.path.basename(dirname), name=name))
    
    if str(request.form.get("submit")) == "Download your csv file":
        return send_from_directory(directory=HERE + "/datasets", path="data.csv")

    """
    if str(request.form.get("submit")) == "Get Information":
        return redirect(url_for("data_profiling",
                                name=name,
                                dirname=dirname,
                                algorithm=algorithm,
                                wants_support=wants_support,

                                support=support,
                                confidence=confidence
                                )
                        )
    """
    """

    if str(request.form.get("submit")) == "Apply modifications":
         
        return redirect(url_for('apply_modifications', 
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                        wants_support=wants_support,
                    
                        support=support,
                        confidence=confidence))
    """

    # else:

    return render_template("home/profiling.html",
                           name=name,
                           dirname=dirname,
                           columns=columns,
                           algorithm=algorithm,
                           average_rank=average_rank,
                           techniques=techniques,
                           best_imputation_method=best_imputation_method(),
                           information_to_print=information_to_print,

                           support=support,
                           confidence=confidence,

                           typeList=typeList,
                           typeNUMlist=typeNUMlist,
                           wants_support=wants_support,

                           min_values=min_values,
                           max_values=max_values,
                           medianList=medianList,
                           uniqueness=uniqueness,
                           accuracy=accuracy,
                           completeness=completeness,

                           profile=profile,
                           distr_html_list=distr_html_list,
                           distrCAT_html_list=distrCAT_html_list,
                           outliers_html_list=outliers_html_list,
                           risp_missing_val=risp_missing_val,
                           risp_correlation=risp_correlation,
                           risp_outliers=risp_outliers,
                           risp_description=risp_description,
                           risp_description_s=get_schematic_description(),
                           alertMessage=alertMessage,
                           exp2_summary=exp2_summary,
                           exp1=exp1,
                           summ_corr=summ_corr,
                           exp2=best_imputation_method1()
                           )



"""
@app.route('/apply_modifications/<name>/<dirname>/<algorithm>/<support>/<confidence>/<wants_support>/', methods=['GET', 'POST'])
def apply_modifications(name,dirname,algorithm,support,confidence,wants_support):
               
    
    return redirect(url_for("data_profiling",
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                        wants_support=wants_support,
                    
                        support=support,
                        confidence=confidence
                            )
                        )
"""

@app.route("/get_information/<name>/<dirname>/<algorithm>/<support>/<confidence>/<wants_support>", methods=["POST"])
def get_information(name,dirname,algorithm,support,confidence,wants_support):
    print("Information requested!")

    global information_to_print
    # information_to_print = "New Information !"

    sorted_list = request.get_json()

    print("method selected: " + sorted_list[0])
    method = sorted_list[0]

    # query the kb to retrieve the requested information

    URI = "neo4j://localhost"
    AUTH = ("neo4j", "ciaociao")

    driver = GraphDatabase.driver(URI, auth=AUTH)

    driver.verify_connectivity()
    print("verify connectivity tutto ok\n")

    # I query what is the technique for that method

    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE)-[:IMPLEMENTED_WITH]->(m:DATA_PREPARATION_METHOD) \
         WHERE m.name = $method \
         RETURN t.name as technique",
        method=method,
        database_="neo4j",
    )

    technique = records[0]
    technique = technique["technique"]
    # print(technique)

    # dimensions affected
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})-[a:AFFECTS]->(d:DQ_DIMENSION) \
         RETURN d.name AS dimension, a.influence_type AS influence_type ",
        technique=technique,
        database_="neo4j",
    )

    dimensions = []
    dimensions_influences = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        dimensions.append(record["dimension"])
        dimensions_influences.append(record["influence_type"])
    # print(dimensions)
    # print(dimensions_influences)

    # ml affected
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})<-[:BENEFITS_FROM]-(ml:ML_APPLICATION) \
         RETURN ml.application_method AS ml_algorithm",
        technique=technique,
        database_="neo4j",
    )

    ml_algorithms = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        ml_algorithms.append(record["ml_algorithm"])
    # print(ml_algorithms)

    # depends on feature, technique
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_TECHNIQUE{name: $technique})-[d:DEPENDS_ON]->(p:DATA_PROFILE_FEATURE) \
        RETURN p.name AS feature, d.description AS description ",
        technique=technique,
        database_="neo4j",
    )

    t_features = []
    t_features_descriptions = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        t_features.append(record["feature"])
        t_features_descriptions.append(record["description"])
    # print(t_features)
    # print(t_features_descriptions)

    # depends on feature, method
    records, summary, keys = driver.execute_query(
        "MATCH (t:DATA_PREPARATION_METHOD{name: $method})-[d:DEPENDS_ON]->(p:DATA_PROFILE_FEATURE) \
        RETURN p.name AS feature, d.description AS description",
        method=method,
        database_="neo4j",
    )

    m_features = []
    m_features_descriptions = []

    # Loop through results and do something with them
    for record in records:
        # print(record)
        m_features.append(record["feature"])
        m_features_descriptions.append(record["description"])
    # print(m_features)
    # print(m_features_descriptions)

    text = "TECHNIQUE " + technique + " :<br/>"

    for dimension, dimension_influence in zip(dimensions, dimensions_influences):
        text = text + "Affects the dimension \"" + dimension + "\" in this way: " + dimension_influence + ".<br/>"

    for ml_algorithm in ml_algorithms:
        text = text + "The ML algorithm \"" + ml_algorithm + "\" benefits from this technique" + ".<br/>"

    for t_feature, t_feature_description in zip(t_features, t_features_descriptions):
        text = text + "Depends on the dataset feature \"" + t_feature + "\" in this way: " + t_feature_description + "<br/>"

    text = text + "Method \"" + method + "\" is a possible implementation method for this technique" + ".<br/>"

    for m_feature, m_feature_description in zip(m_features, m_features_descriptions):
        text = text + "This method depends on the dataset feature \"" + m_feature + "\" in this way: " + m_feature_description + "<br/>"

    print(text)
    text = Markup(text)

    information_to_print = text

    driver.close()

    return redirect(url_for("data_profiling",
                        name=name,
                        dirname=dirname,
                        algorithm=algorithm,
                        wants_support=wants_support,

                        support=support,
                        confidence=confidence
                        )
                )


def imputation_name_conversion(c_name):
    if c_name == "no_impute":
        return "No Imputation"
    if c_name == "impute_standard":
        return "Standard Value Imputation"
    if c_name == "impute_mean":
        return "Mean Imputation"
    if c_name == "impute_std":
        return "Std Imputation"
    if c_name == "impute_mode":
        return "Mode Imputation"
    if c_name == "impute_median":
        return "Median Imputation"
    if c_name == "impute_knn":
        return "KNN Imputation"
    if c_name == "impute_mice":
        return "Mice Imputation"
    if c_name == "impute_soft":
        return "softImpute Imputation"
    if c_name == "impute_random":
        return "Random Imputation"
    if c_name == "impute_linear_and_logistic":
        return "Linear and Logistic Regression Imputation"
    if c_name == "impute_linear_regression":
        return "Linear Regression Imputation"
    if c_name == "impute_logistic_regression":
        return "Logistic Regression Imputation"
    return "Name not found"













