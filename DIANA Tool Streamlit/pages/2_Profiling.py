from cgitb import small
import os
import webbrowser

import navigation as navigation
import streamlit as st
import time
import pandas as pd
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
import streamlit_nested_layout
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import missingno as msno
import matplotlib.pyplot as plt
from io import BytesIO


st.set_page_config(page_title="Profiling", layout="wide", initial_sidebar_state="collapsed")


# def app():

df = st.session_state['df']
# st.write(df.head())

st.markdown(
    """
    <style>
    .css-1gb49b3 {
        position: absolute;
        top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
    border: 1px solid black;
    border-radius: 5px;
    padding: 10px;
    font-size: 2rem;
    color: black; 
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

st.markdown(
    """
    <style>
    .css-145kmo2 {
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<style>h1{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)



numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

pages = {
    "Dataset": "pagina_1",
    "Data Summary": "pagina_2",
    "BoxPlot": "pagina_3",
    "Heatmap" : "pagina_4",
    "Distribution Plot" : "pagina_5",
    "Distribution Categorical" : "pagina_6",
    "Data Missing" : "pagina_7"
}

# Crea un menu a tendina nella barra laterale
scelta_pagina = st.sidebar.selectbox("Select a page:", list(pages.keys()))


if scelta_pagina == 'Dataset':
    st.title("Dataset")
    st.write(df)
    st.write("---")

# Sottopagina 1
# Creazione della tabella in Streamlit con sfondo bianco
elif scelta_pagina == "Data Summary":
    st.markdown("""
        <style>
        .stApp {
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("Data Summary:")
    num_selected_variables = len(df.columns)
    num_rows = len(df)
    missing_cells = df.isnull().sum().sum()
    vars_with_missing_cells = (df.isnull().sum() > 0).sum()
    num_duplicates = df.duplicated().sum()

    # Conta il numero di variabili numeriche e categoriche
    numeric_vars = df.select_dtypes(include=['int64', 'float64']).shape[1]
    categorical_vars = df.select_dtypes(include=['object']).shape[1]

    # Creazione del dizionario con i dati
    data = {
        'Number of selected variables': num_selected_variables,
        'Number of rows/observations': num_rows,
        'Missing Cells': missing_cells,
        'Variables with missing cells': vars_with_missing_cells,
        'Duplicate rows': num_duplicates,
        'Numeric Variables': numeric_vars,
        'Categorical Variables': categorical_vars
    }
    data_df = pd.DataFrame(data.items(), columns=["Metric", "Value"])
    st.dataframe(data_df.style.set_properties(**{'background-color': 'white'}), width=400)


elif scelta_pagina == "BoxPlot":
# Crea un'app Streamlit
    st.title("BoxPlot")

# Aggiungi il DataFrame alle opzioni per la scelta delle colonne
    selected_columns = st.multiselect("Select column for Box Plots", numeric_cols)

# Verifica che almeno una colonna sia stata selezionata
    if len(selected_columns) > 0:
    # Creazione della directory per i grafici se non esiste
        os.makedirs("box_plots", exist_ok=True)

        for col in selected_columns:
        # Crea un grafico a scatola per ciascuna colonna selezionata
            fig = go.Figure(data=go.Box(x=df[col], boxpoints="outliers"))
            fig.update_layout(
                title=f"Box Plot of {col}",
                height=300,
                showlegend=False,
                margin=dict(l=5, r=10, b=5, t=30),
                paper_bgcolor="#ffffff",
                font=dict(color='#555'),
                plot_bgcolor="#e6e6fa",
            )

        # Salva il grafico come file HTML
            filename = f"box_plots/{col}_box_plot.html"
            fig.write_html(filename)

        # Mostra il grafico su Streamlit
            st.write(f"Box Plot of {col}")
            st.plotly_chart(fig)

    st.write("---")

elif scelta_pagina == "Heatmap":
    df_numeric = df.select_dtypes(include=['int64', 'float64'])

    df_numeric=df_numeric.dropna()

    st.title("Heatmap")
    @st.cache_resource
    def compute_correlation_matrix(df):
        correlation_matrix = df.corr(method='spearman')
        return correlation_matrix

# All'interno della funzione principale di Streamlit
    correlation_matrix = compute_correlation_matrix(df_numeric)
    correlation_matrix = round(correlation_matrix, 2)

# Crea una lista di nomi di colonne
    column_names = list(correlation_matrix.columns)

# Crea il grafico di tipo "heatmap" con Plotly Figure Factory
    heatmap_fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=column_names,
        y=column_names,
        colorscale="YlOrRd",
        annotation_text=correlation_matrix.values,
        showscale=True
    )

# Personalizza il layout del grafico
    heatmap_fig.update_layout(
        title="Heatmap",
        xaxis=dict(side="top"),
        margin=dict(l=5, r=10, b=5, t=30),
        paper_bgcolor="#ffffff",  # Modalità chiara
        font=dict(color="#555"),
        plot_bgcolor="#e6e6fa",  # Modalità chiara
    )

# Mostra il grafico nella pagina Streamlit
    st.plotly_chart(heatmap_fig)

    st.write("---")

elif scelta_pagina == "Distribution Plot":
    st.title("Distribution Plot")

# Barra di selezione delle colonne
    selected_column = st.selectbox("Select a column", df.select_dtypes(include=['int64', 'float64']).columns)

# Filtra i valori NaN e infiniti per la colonna selezionata
    data = df[selected_column][~df[selected_column].isin([float('nan'), float('inf'), -float('inf')])]

    statistics = {
        "Minimum": data.min(),
        "Maximum": data.max(),
        "Mean": data.mean(),
        "Median": data.median(),
        "Variance": data.var(),
        "Standard Deviation": data.std(),
        "Distinct Values": data.nunique(),
        "Missing Values": data.isnull().sum(),
        "Zero Values": (data == 0).sum(),
    }

    # Show statistics in horizontal tables
    col1, col2, col3, col4, col5,col6 = st.columns(6)
    with col1:
        st.write("Column Statistics:")
        st.table(pd.DataFrame(statistics.items(), columns=["Metric", "Value"]).set_index("Metric"))

    with col2:
    # Show Common Values
        st.write("Common Values:")
        common_values = data.value_counts().head(10)
        st.table(pd.DataFrame({"Value": common_values.index, "Count": common_values.values}).set_index("Value"))

    with col3:
    # Show Extreme Values (Min)
        st.write("Extreme Values (Min):")
        extreme_min_values = data.nsmallest(10)
        st.table(pd.DataFrame({"Value": extreme_min_values.index, "Count": extreme_min_values.values}).set_index("Value"))

    with col4:
    # Show Extreme Values (Max)
        st.write("Extreme Values (Max):")
        extreme_max_values = data.nlargest(10)
        st.table(pd.DataFrame({"Value": extreme_max_values.index, "Count": extreme_max_values.values}).set_index("Value"))

# Crea un grafico di distribuzione usando Plotly Express
    fig = ff.create_distplot([data], [selected_column], show_hist=False)

# Aggiorna il layout del grafico
    fig.update_layout(xaxis_title='Value', yaxis_title='Density', width=600, height=400)

# Mostra il grafico nella pagina Streamlit
    st.plotly_chart(fig)

    st.write("---")

elif scelta_pagina == "Distribution Categorical" :

    st.title("Distribution Categorical")

# Lista delle colonne categoriche
    typeCATlist = [col for col in df.columns if df[col].dtype == 'object']

# Seleziona una colonna dalla lista tramite selectbox
    selected_col = st.selectbox("Select a categorical column", typeCATlist)

# Filtra il DataFrame in base alla colonna selezionata
    filtered_df = df[df[selected_col].notna()]

# Calcola il conteggio delle categorie
    vc = filtered_df[selected_col].value_counts()
    vc_df = pd.DataFrame({'var': vc.index, 'count': vc.values})
    st.dataframe(vc_df)

# Crea il grafico a barre
    fig = px.bar(vc_df, x='var', y='count')

# Personalizza il layout del grafico
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Count',
        title=f'Categorical Count for Column {selected_col}',
        width=600,
        height=400,
        margin=dict(l=5, r=10, b=5, t=30),
        paper_bgcolor="#ffffff",  # Modalità chiara
        font=dict(color="#555"),
        plot_bgcolor="#e6e6fa",  # Modalità chiara
    )

# Mostra il grafico nella pagina Streamlit
    st.plotly_chart(fig)
    st.write("---")

elif scelta_pagina == "Data Missing" :
    df_copy = df.copy()



    st.title("Data Missing")

# Matrice dei dati mancanti
   # st.subheader("Data missing matrix")
    #fig_matrix = msno.matrix(df, color=(0.329, 0.059, 0.286), fontsize=12)
    #buffered = BytesIO()
    #fig_matrix.get_figure().savefig(buffered, format="png")
    #st.image(buffered)

# Calculate and display the percentage of missing values per column
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    st.subheader("Percentage of Missing Values per Column")
    st.write(missing_percentage)

# Grafico a barre per i dati mancanti
    st.subheader("Distribution graph for data missing")
    fig_bars = px.bar(df.isnull().sum(), x=df.columns, y=df.isnull().sum(), text=df.isnull().sum(),
                  color_continuous_scale='bupu')
    fig_bars.update_layout(margin=dict(l=5, r=10, b=5, t=30))
    fig_bars.update_layout(paper_bgcolor="#ffffff", font=dict(color='#555'), plot_bgcolor="#e6e6fa")
    st.plotly_chart(fig_bars)

# Heatmap dei dati mancanti
    #st.subheader("Data missing heatmap")
    #df_copy = df_copy.iloc[:, [i for i, n in enumerate(np.var(df_copy.isnull(), axis='rows')) if n > 0]]
    #corr_mat = df_copy.isnull().corr()
    #corr_mat = round(corr_mat, 2)
    #fig_heat = px.imshow(corr_mat, x=corr_mat.index, y=corr_mat.columns, aspect="auto", text_auto=True,
    #                    color_continuous_scale='bupu')
    #fig_heat.update_xaxes(side="top")
    #fig_heat.update_layout(autosize=True, margin=dict(l=5, r=10, b=5, t=30))
    #fig_heat.update_layout(paper_bgcolor="#ffffff", font=dict(color='#555'), plot_bgcolor="#e6e6fa")
    #st.plotly_chart(fig_heat)

    st.write("---")

if st.button("Continue", key="continue_cleaning"):
    switch_page("Transformation")