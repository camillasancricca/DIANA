from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import pandas as pd
import functions
import time
import openai
import json
import re
import time



app = Flask(__name__, template_folder='/Users/martinacaffagnini/Desktop/gpt', static_folder='/static')
app.secret_key = 'your_secret_key'
openai.api_key = "key"
fixed_question = "Hi! I will provide you with a JSON file of a certain dataset. I would like you to return me a description of the dataset and a description of each column. Additionally, I would like you to provide me with 8 total use cases of which two for clustering, two for classification, two for time series analysis and two for regression. The description of each use case should be very schematic: name of the target column of the analysis, name of the required analysis and a brief description in non technical language and very easy to understand even by non expert users without naming algorithms directly but only what the use case does. I would like all the answers to be given directly in a code-like structure. I want the description of the dataset in a variable named Description, the descriptions of the variables in a dictionary like structure named Variables with the name of the variable and the corresponding description and the use cases in a dictionary structure named Use_cases containing a column for the title of the use case that must summarize what the analysis is for named Title, a column for the target variable named Target_variable, a column for the description named Description and a column with the name of the type of analysis required named Analysis. Please provide all dictionaries in such a way that each element is written on a sigle row. Thank you."


class Results:
    def __init__(self, descrizione, sample=None, use_cases=None, variables=None):
        self.descrizione = descrizione
        self.sample = sample
        self.use_cases = use_cases
        self.variables = variables

    def set_use_cases(self, use_cases):
        self.use_cases = use_cases

    def set_sample(self, sample):
        self.sample = sample

    def set_variables(self, variables):
        self.variables = variables

    def set_description(self, description):
        self.descrizione = description

class Analysis:
    def __init__(self, analysis, target, description):
        self.analysis = analysis
        self.target = target
        self.description = description

    def set_analysis(self, analysis):
        self.analysis = analysis

    def set_target(self, target):
        self.target = target

    def set_description(self, description):
        self.description = description

'''
@app.route('/')
def landing_page():
    descr = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/descr.csv')

    # Crea una nuova struttura di dati come richiesto
    new_data = []
    for index, row in descr.iterrows():
        new_entry = {
            "activities": row['activities'],
            "descriptions": row['descriptions']
        }
        new_data.append(new_entry)

    return render_template('boh.html', dataframe=new_data)'''

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # Esempio di caricamento del file CSV
        uploaded_file = request.files['csv_file']
        if uploaded_file and uploaded_file.filename.endswith('.csv'):
            prova = pd.read_csv(uploaded_file, nrows=5)
            prova.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/'+str(uploaded_file.name)+'.csv', index=False)
            data_json = functions.csv_to_json('/Users/martinacaffagnini/Desktop/gpt/uploads/csv_file.csv', '/Users/martinacaffagnini/Desktop/gpt/uploads/output.json')        
            
            response = functions.ask_question(fixed_question, data_json)
            file_path = '/Users/martinacaffagnini/Desktop/gpt/risposta.txt'
            file_path1 = '/Users/martinacaffagnini/Desktop/gpt/risposta1.txt'
            with open(file_path, 'r') as file:
                text = file.read()
            
            text = re.sub(r'"', "'", text)
            with open(file_path1, 'w') as file:
                file.write(text)

            description, variables_df, use_cases_df = functions.extract_data(file_path1)
            descr = pd.DataFrame()
            d = []
            d.append(description)
            descr['Description']=d
            descr.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/descr.csv', index=False)
            use_cases_df.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/uc.csv', index=False)
            variables_df.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/v.csv', index=False)

            vars = variables_df['Variable'].to_list()
            response_vars = functions.ask_question_vars(vars, data_json)
            result = "Dati processati con successo!"
            return jsonify({'result': result})
        else:
            error = "Carica solo file CSV!"
            return jsonify({'error': error})
    return render_template('landing_page.html')

@app.route('/process_question', methods=['POST', 'GET'])
def process_question():
    selected_option = request.form['option']
    results = session.get('results', {})
    uc = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/uc.csv')
    vdf = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/v.csv')
    d = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/descr.csv')
    descr = d['Description'].to_list()[0]

    def add_button_use_case(row):
        return_string = '<button class="confirmButton" id="'+str(row['Title'])+'">Select</button>'
        #return '<button class="but" onclick="alert(\'Selected use case: \n' + row['Title'] + '\')">Select</button>'
        return return_string

    uc['Choose a use case'] = uc.apply(add_button_use_case, axis=1)

    def add_button_more(row):
        df = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/var_add_info.csv')
        name = row['Variable']
        your_condition = df['name'] == name
        filtered_df = df[your_condition]
        result = filtered_df['info']
        result_list = result.to_list()
        return '<button class="but" onclick="alert(\'More info: \n' + str(result_list[0]) + '\')">Select</button>'

    vdf['More info'] = vdf.apply(add_button_more, axis=1)
    #sample = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/csv_file.csv', sep=';')
    sample = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/csv_file.csv')
    obj = Results(descr, sample, uc, vdf)
    if selected_option == 'option1':
        # Esegui azioni basate sulla risposta option1
        df = obj.sample
        lista_res = df.columns.tolist()
        lista_res.append('NaN')
        lista_analysis = ['Classification', 'Clustering', 'Regression', 'Time Series Analysis', 'Descriptive Analysis']
        return render_template('result1.html', lista_res = lista_res, results = obj, lista_analysis=lista_analysis)
    elif selected_option == 'option2':
        # Esegui azioni basate sulla risposta option2
        return render_template('result2.html', results = obj)
    elif selected_option == 'option3':
        # Esegui azioni basate sulla risposta option3
        return render_template('result3.html', results = obj)


@app.route('/question')
def question():
    return render_template('question_form.html')

@app.route('/result_page')
def result_page():
    results = session.get('results', {})
    csv_result = results.get('csv_result', None)
    question_result = results.get('question_result', None)
    if csv_result is not None and question_result is not None:
        return f"CSV Result: {csv_result}<br>Question Result: {question_result}"
    else:
        return "Nessun risultato disponibile."
    
@app.route('/upload', methods = ['POST', 'GET'])
def handle_text():
    if request.method == 'POST':
        text = request.form['text_data']
        with open('/Users/martinacaffagnini/Desktop/gpt/uploads/output.json', 'r') as file:
            # Parse the JSON data using json.load() or json.loads()
            data = json.load(file)
        response = functions.ask_question_use_case(text, data)
        print(response)
        pattern = r"{(.*?)}"  # Questa regex cattura il testo tra apici singoli.
        risultato = re.findall(pattern, response, re.DOTALL)
        print(risultato)
        pattern = r'"Target Variable": "(.*?)"'  # Questa regex cattura il testo tra apici singoli.
        tmp = re.findall(pattern, risultato[0])
        target_variable = tmp[0]
        pattern = r'"Analysis Type": "(.*?)"'  # Questa regex cattura il testo tra apici singoli.
        tmp1 = re.findall(pattern, risultato[0])
        analysis_type = tmp1[0]
        pattern = r'"Title": "(.*?)"'  # Questa regex cattura il testo tra apici singoli.
        tmp1 = re.findall(pattern, risultato[0])
        title = tmp1[0]
        pattern = r'"Explanation": "(.*?)"'  # Questa regex cattura il testo tra apici singoli.
        tmp1 = re.findall(pattern, risultato[0])
        explanation = tmp1[0]
        response = 'Target Variable: '+str(target_variable)+'\n'+'Analysis Type: '+str(analysis_type)+'\n'+'Title: '+str(title)+'\n'+'Explanation: '+str(explanation)+'\n'
        prova = pd.DataFrame()
        prova['analysis']=analysis_type
        prova['variable']=target_variable
        prova['title']=title  
        prova.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv', index = False)
        return response
    

@app.route('/get_info', methods = ['POST', 'GET'])
def get_info():
    if request.method == 'POST':
        text = str(request.data)
        pattern = r"b'(.*?)'"  # Questa regex cattura il testo tra apici singoli.
        risultati = re.findall(pattern, text)
        for risultato in risultati:
            title = risultato
        scenarios = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/uc.csv')
        scenarios = scenarios.loc[scenarios['Title'] == title]
        target = str(scenarios['Target Variable'].to_list()[0])
        analysis = str(scenarios['Analysis Type'].to_list()[0])
        result = str('Title: '+title+'<br>'+'Target Variable: '+target+'<br>'+'Analysis Type: '+analysis+'<br>')
    return result

@app.route('/invia-dati', methods=['POST'])
def invia_dati():
    text = str(request.data)
    pattern = r"b'(.*?)'"  # Questa regex cattura il testo tra apici singoli.
    risultato = re.findall(pattern, text)
    #tmp = risultato[0].split('=')
    #tmp1 = tmp[1]
    activities = risultato[0].split('%2C')
    a=[]
    d=[]
    for element in activities:
        tmp2 = element.split('%20')
        res=''
        for e in tmp2:
            res = res+e+' '
        a.append(res)
        descr = functions.ask_question_description(res)
        d.append(descr)
    descr = pd.DataFrame()
    descr['activities']=a
    descr['descriptions']=d
    descr.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/descr.csv', index=False)
    return jsonify({'message': 'Dati ricevuti con successo'})

@app.route('/pagina_task', methods=['POST', 'GET'])
def pagina_task():
    descr = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/descr.csv')

    # Crea una nuova struttura di dati come richiesto
    new_data = []
    for index, row in descr.iterrows():
        new_entry = {
            "activities": row['activities'],
            "descriptions": row['descriptions']
        }
        new_data.append(new_entry)

    return render_template('task.html', dataframe=new_data)

@app.route('/submit_choice', methods=['POST'])
def submit_choice():
    # Ricevi i dati inviati dal form
    selected_card_index = int(request.form.get('cardIndex'))
    selected_choice = request.form.get('choice')

    # Ora puoi fare qualcosa con i dati, come salvare la scelta in una base dati

    return jsonify({"message": "Dati ricevuti con successo!"})

@app.route('/receive_dati', methods=['POST', 'GET'])
def receive_dati():
    if request.method == 'POST':
        text = str(request.data)
        pattern = r"b'(.*?)'"  # Questa regex cattura il testo tra apici singoli.
        risultati = re.findall(pattern, text)
        data = json.loads(risultati[0])
        df = pd.DataFrame([data])        
        df.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv', index = False)
    
    res = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv')
    AnalysisObj = Analysis(res['analysis'].to_list()[0], res['variable'].to_list()[0], res['title'].to_list()[0])
        
    return render_template('continue.html', results = AnalysisObj)

@app.route('/prova', methods=['GET', 'POST'])
def prova():
    if request.method == 'POST':
        text = str(request.data)
        pattern = r"b'(.*?)'"  # Questa regex cattura il testo tra apici singoli.
        risultato = re.findall(pattern, text)
        pattern = r'{"variabileGlobale":"(.*?)"}'  # Questa regex cattura il testo tra apici singoli.
        risultato = re.findall(pattern, text)
        print(risultato[0])
        pattern = r'Title: (.*?)<br>'  # Questa regex cattura il testo tra apici singoli.
        tmp = re.findall(pattern, risultato[0])
        title = tmp[0]
        t = []
        t.append(title)
        pattern = r'Target Variable: (.*?)<br>'  # Questa regex cattura il testo tra apici singoli.
        tmp1 = re.findall(pattern, risultato[0])
        target = tmp1[0]
        tar = []
        tar.append(target)
        pattern = r'Analysis Type: (.*?)<br>'  # Questa regex cattura il testo tra apici singoli.
        tmp2 = re.findall(pattern, risultato[0])
        analysis = tmp2[0]
        a = []
        a.append(analysis)
        prova = pd.DataFrame()
        prova['analysis']=a
        prova['variable']=tar
        prova['title']=t   
        prova.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv', index = False)

    res = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv')
    AnalysisObj = Analysis(res['analysis'].to_list()[0], res['variable'].to_list()[0], res['title'].to_list()[0])
    if (res['analysis'].to_list()[0] == 'Time Series Analysis') | (res['analysis'].to_list()[0] == 'Time series analysis'):
        return render_template('continue1.html', results = AnalysisObj)
    else:
        return render_template('continue.html', results = AnalysisObj)
    
@app.route('/automated')
def automated():
    df = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv')
    analysis = df['analysis'].to_list()[0]
    if analysis == 'Clustering':
        return render_template('clustering.html')
    if analysis == 'Regression':
        return render_template('regression.html')
    if analysis == 'Classification':
        return render_template('classification.html')
    if analysis == 'Exploratory Analysis':
        return render_template('descriptive.html')

@app.route('/automated_ts')
def automated_ts():
    return render_template('time_series.html')

@app.route('/analysis')
def analysis():
    df = pd.read_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/analysis.csv')
    analysis = df['analysis'].to_list()[0]
    print(analysis)
    if (analysis == 'Clustering') | (analysis == 'Unsupervised Clustering'):
        return render_template('clustering.html')
    if analysis == 'Regression':
        return render_template('regression.html')
    if analysis == 'Classification':
        return render_template('classification.html')
    if (analysis == 'Time Series Analysis') | (analysis == 'Time series analysis'):
        return render_template('time_series.html')
    if analysis == 'Exploratory Analysis':
        return render_template('descriptive.html')

if __name__ == '__main__':
    app.run(debug=True)
