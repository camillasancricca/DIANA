import csv
import json
from flask import Flask, request, jsonify
import openai
import re
import ast
import pandas as pd

def csv_to_json(csv_file, json_file):
    # Initialize an empty list to store the data
    data = []

    # Open the CSV file for reading
    with open(csv_file, 'r') as csv_file:
        # Create a CSV reader
        csv_reader = csv.DictReader(csv_file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Append the row as a dictionary to the data list
            data.append(row)

    # Open the JSON file for writing
    with open(json_file, 'w') as json_file:
        # Write the data list to the JSON file
        json.dump(data, json_file, indent=4)
    return data

def ask_question(fixed_question, user_input):
    system_message = "You are a useful assistant who anderstands data science and can reason with data."
    user_message = str(fixed_question)+str(user_input)
    # Componi la domanda utilizzando la domanda fissa e il parametro variabile
    #question = f"{fixed_question} {user_input}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : system_message},
    {"role": "user", "content" : user_message}])
    
    status_code = response['choices'][0]['finish_reason']
    result = response['choices'][0]['message']['content']
    
    # Salva la risposta in un file di testo
    with open('/Users/martinacaffagnini/Desktop/gpt/risposta.txt', 'w') as file:
        file.write(result)

    return result


def ask_question_vars(vars, json_file):
    var_df = pd.DataFrame()
    name=[]
    info=[]

    system_message = "You are a useful assistant who anderstands data science and can reason with data."
    question = 'Can you please provide me with a more in depth explanation of this variable? Please answer in 50 words or less. Consider the dataset provided in json to answer:'+str(json_file)
    for var in vars:
        user_message = str(question)+str(var)
        # Componi la domanda utilizzando la domanda fissa e il parametro variabile
        #question = f"{fixed_question} {user_input}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role": "system", "content" : system_message},
        {"role": "user", "content" : user_message}])
    
        status_code = response['choices'][0]['finish_reason']
        result = response['choices'][0]['message']['content']

        name.append(var)
        result = re.sub(r'"', "'", result)
        result = re.sub(r',', '', result)
        result = re.sub(r"'", '', result)



        info.append(result)
    var_df['name']=name
    var_df['info']=info
    var_df.to_csv('/Users/martinacaffagnini/Desktop/gpt/uploads/var_add_info.csv', index=False)
    return var_df


def ask_question_use_case(text, j):
    system_message = "You are a useful assistant who anderstands data science and can reason with data."
    question = 'Given the following dataset, can you identify the target variable and analysis method that corresponds to the following description? I want the answer to be structured as follows: {"Target Variable":"informations to add", "Analysis Type": "information to add", "Title": "information to add", "Explanation": "explanation"}. The variable named "Target Variable" for the target variable and a variable named "Analysis Type" for the analysis type. Concerning the analysis type, you should select one among Classification, Regression, CLustering, Time Series Analysis or Exploratory Analysis. Can you also provide me with a brief title for the provided scenario called "Title". Also, in a variable named "Explanation", can you describe the reasons behind yur choices.'
    
    user_message = str(question)+str(j)+str(text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : system_message},
    {"role": "user", "content" : user_message}])

    status_code = response['choices'][0]['finish_reason']
    result = response['choices'][0]['message']['content']
    
    with open('/Users/martinacaffagnini/Desktop/gpt/use_case.txt', 'w') as file:
        file.write(result)
        
    return result




def extract_data(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    # Estrai la variabile "Description"
    text = re.sub(r'"', "'", text)
    pattern = r"Description\s*=\s*'([^']*)'"

    # Use re.search to find the match
    match = re.search(pattern, text)

    if match:
        description_initial = match.group(1)
    else:
        description_initial = "ciao"


    # Estrai il dizionario delle variabili
    variables_match = re.search(r'Variables = {(.*?)}', text, re.DOTALL)
    if variables_match:
        variables_text = variables_match.group(1)
        variables = re.findall(r"'([^']+)': '([^']+)'", variables_text)
    else:
        variables = []

    # Estrai la lista dei casi d'uso
    '''use_cases_match = re.search(r'Use_cases = \[(.*?)\]', text, re.DOTALL)
    #use_cases_match = re.search(r'Use_cases\s*=\s*\[(.*?)\]', text, re.DOTALL)
    #use_cases_match = re.search(r"Use_cases\s*=\s*\[\s*{([^}]+)}\s*]", text, re.DOTALL)
    if use_cases_match:
        use_cases_text = use_cases_match.group(1)
        use_cases = re.findall(r"{'title': '(.*?)',\s* 'target_variable': '(.*?)',\s* 'description': '(.*?)',\s* 'analysis_type': '(.*?)'}", use_cases_text)
    else:
        use_cases = []
    '''
    cases_match = re.search(r'Use_cases = \[(.*?)\]', text, re.DOTALL)
    if cases_match:
        pattern = r"\s*{\s*'Title': '([^']*)'\s*,\s*'Target_variable': '([^']*)'\s*,\s*'Description': '([^']*)'\s*,\s*'Analysis': '([^']*)'\s*},*"
        use_cases = cases_match.group(1)
        matches = re.findall(pattern, use_cases)
    else:
        use_cases = []

    # Utilizziamo findall per trovare tutte le corrispondenze
    tit =[]
    target = []
    descr = []
    type = []
    if matches:
        for match in matches:
            title = match[0]
            target_variable = match[1]
            description = match[2]
            analysis_type = match[3]
            tit.append(title)
            target.append(target_variable)
            descr.append(description)
            type.append(analysis_type)
    else:
        print("Nessuna corrispondenza trovata.")

        '''use_cases = cases_match.group(1)
        target = [item['Target_variable'] for item in use_cases]
        tit = [item['Title'] for item in use_cases]
        descr = [item['Description'] for item in use_cases]
        type = [item['Analysis'] for item in use_cases]'''


    # Crea un DataFrame per le variabili
    variables_df = pd.DataFrame(variables, columns=['Variable', 'Description'])

    # Crea un DataFrame per i casi d'uso
    use_cases_df = pd.DataFrame()
    use_cases_df['Title'] = tit
    use_cases_df['Target Variable'] = target
    use_cases_df['Description'] = descr
    use_cases_df['Analysis Type']= type

    return description_initial, variables_df, use_cases_df


def ask_question_description(text):
    system_message = "You are a useful assistant who anderstands data science and can reason with data."
    question = 'Can you give me a simple description of the following data preparation task in around 50 words or less. The task to describe is:'
    
    user_message = str(question)+str(text)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content" : system_message},
    {"role": "user", "content" : user_message}])

    status_code = response['choices'][0]['finish_reason']
    result = response['choices'][0]['message']['content']
    return result

