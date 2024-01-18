import dash
from dash import dcc, html
import pandas as pd
import openai
import plotly.express as px
from dash.dependencies import Input, Output

# Inizializza l'applicazione Dash
app = dash.Dash(__name__)

# Imposta la tua chiave API GPT-3
openai.api_key = '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'

# Carica il dataset da un file CSV locale
df = pd.read_csv("beers.csv")  # Assicurati che il file sia nella stessa directory del tuo script

# Crea il layout dell'app
app.layout = html.Div([
    html.H1("Distribuzione degli Attributi del Dataset"),
    html.Div(id='plots-and-explanations'),
])

# Crea una funzione per ottenere la spiegazione
def get_gpt_explanation(column_name, filtered_df):
    # Chiedi una spiegazione a ChatGPT basata sui dati filtrati per una specifica colonna
    gpt_response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Explain the data  which comes from zooming in on the original density chart, keeping in mind that the explanation should be easily understandable for a non-expert user and contextualize the data obtained from the zoom:\n{filtered_df.head()}",
        max_tokens=1000
    )
    return gpt_response.choices[0].text

# Crea i grafici e le spiegazioni per ciascuna colonna del dataset
plots_and_explanations = []
for column in df.columns:
    fig = px.histogram(df, x=column, title=f"Distribuzione di '{column}'")
    plots_and_explanations.append(html.Div([
        html.H3(f"Distribuzione di '{column}'"),
        dcc.Graph(figure=fig, id=f'plot-{column}'),
        html.Div(id=f'explanation-{column}')
    ]))

app.layout['plots-and-explanations'].children = plots_and_explanations

@app.callback(
    Output('plots-and-explanations', 'children'),
    [Input(f'plot-{column}', 'relayoutData') for column in df.columns]
)
def update_plots_and_explanations(*args):
    figs_and_explanations = []

    for column, relayout_data in zip(df.columns, args):
        if relayout_data:
            if pd.api.types.is_numeric_dtype(df[column]):
                x_range = [relayout_data.get('xaxis.range[0]', df[column].min()),
                           relayout_data.get('xaxis.range[1]', df[column].max())]

                # Esegui la tua analisi e ottieni i nuovi dati in base allo zoom
                filtered_df = df[(df[column] >= x_range[0]) & (df[column] <= x_range[1])]

                # Chiedi una spiegazione a ChatGPT
                explanation = get_gpt_explanation(column, filtered_df)

                # Restituisci il risultato dello zoom e la spiegazione di ChatGPT
                fig = px.histogram(filtered_df, x=column, title=f"Distribuzione di '{column}'")
                figs_and_explanations.extend([html.Div([
                    html.H3(f"Distribuzione di '{column}'"),
                    dcc.Graph(figure=fig, id=f'plot-{column}'),
                    html.Div(explanation)
                ])])
            else:
                fig = px.histogram(df, x=column, title=f"Distribuzione di '{column}'")
                figs_and_explanations.extend([html.Div([
                    html.H3(f"Distribuzione di '{column}'"),
                    dcc.Graph(figure=fig, id=f'plot-{column}'),
                    html.Div('')
                ])])
        else:
            fig = px.histogram(df, x=column, title=f"Distribuzione di '{column}'")
            figs_and_explanations.extend([html.Div([
                html.H3(f"Distribuzione di '{column}'"),
                dcc.Graph(figure=fig, id=f'plot-{column}'),
                html.Div('')
            ])])
    return figs_and_explanations


if __name__ == '__main__':
    app.run_server(debug=True)