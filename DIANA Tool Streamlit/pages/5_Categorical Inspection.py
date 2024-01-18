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
if 'my_dataframe' not in st.session_state:
    st.session_state.my_dataframe = df
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

st.title("Categorical Inspection")
typeCATlist = [col for col in st.session_state.my_dataframe.columns if st.session_state.my_dataframe[col].dtype == 'object']

# Seleziona una colonna dalla lista tramite selectbox
selected_columns = st.selectbox("Select a categorical column", typeCATlist)

# Dizionario per le stringhe valide
valid_strings = st.session_state.setdefault("strings",{})

if selected_columns not in valid_strings:
    valid_strings[selected_columns] = []

# Crea un grafico a scatola per ciascuna colonna selezionata

# Filtra il DataFrame in base alla colonna selezionata

# Calcola il conteggio delle categorie
vc = st.session_state.my_dataframe[selected_columns].value_counts()
vc_df = pd.DataFrame({'var': vc.index, 'count': vc.values})
st.dataframe(vc_df)

# Crea il grafico a barre
fig = px.bar(vc_df, x='var', y='count')

# Personalizza il layout del grafico
fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Count',
    title=f'Categorical Count for Column {selected_columns}',
    width=600,
    height=400,
    margin=dict(l=5, r=10, b=5, t=30),
    paper_bgcolor="#ffffff",  # Modalità chiara
    font=dict(color="#555"),
    plot_bgcolor="#e6e6fa",  # Modalità chiara
)

# Mostra il grafico nella pagina Streamlit
st.plotly_chart(fig)

# Lista delle stringhe valide
st.write(f"Set the valid strings for {selected_columns}:")

valid_values = st.text_input(f"Valid values for {selected_columns}")

if st.button("Add Valid String"):
    if selected_columns not in valid_strings:
        valid_strings[selected_columns] = []
    valid_strings[selected_columns].append(valid_values)


# Mostra le stringhe valide
st.write(f"Valid strings selected for {selected_columns}:")
st.write(valid_strings[selected_columns])

# Ora puoi utilizzare il dizionario 'valid_strings' per accedere alle stringhe valide selezionate per ogni colonna.
st.write("Valid strings selected:")
st.write(valid_strings)

st.write("---")

if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Functional_Dependencies")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Outliers Inspection")