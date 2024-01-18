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

#m = st.markdown("""
#<style>
#div.stButton > button:first-child {
#    background-color: rgb(255, 254, 239);
#    height:6em;
#    width:6em;
#}
#</style>""", unsafe_allow_html=True)

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


st.title("Outliers Inspection")
st.write("Write the range of values. Anything outside of it will be considered an outlier")

numeric_cols = st.session_state.my_dataframe.select_dtypes(include=['int64', 'float64']).columns

# Scelta delle colonne per i BoxPlots
selected_columns = st.multiselect("Select columns for Box Plots", numeric_cols)

# Dizionario per gli intervalli
intervals = st.session_state.setdefault("intervals", {})




# Verifica che almeno una colonna sia stata selezionata
if len(selected_columns) > 0:
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

        # Mostra il grafico su Streamlit
        st.write(f"Box Plot of {col}")
        st.plotly_chart(fig)
        # Input fields per gli intervalli
        st.write(f"Set the interval for {col}:")




        if col in intervals:
            # Se l'intervallo è stato specificato in precedenza, mostra i valori attuali
            min_val = st.number_input(f"Minimum value for {col}", value=intervals[col][0], key=f"min_{col}")
            max_val = st.number_input(f"Maximum value for {col}", value=intervals[col][1], key=f"max_{col}")
        else:
            # Se l'intervallo non è mai stato specificato, mostra input fields vuoti
            min_val = st.number_input(f"Minimum value for {col}", value=st.session_state.my_dataframe[col].min(), key=f"min_{col}")
            max_val = st.number_input(f"Maximum value for {col}", value=st.session_state.my_dataframe[col].max(), key=f"max_{col}")

        # Salva gli intervalli nel dizionario
        intervals[col] = (min_val, max_val)

        st.write("Intervals selected:")
        st.write(intervals[col])

else:
    for colonna in numeric_cols:
        min_val = st.session_state.my_dataframe[colonna].min()
        max_val = st.session_state.my_dataframe[colonna].max()
        intervals[colonna] = (min_val, max_val)

# Ora puoi utilizzare il dizionario 'intervals' per accedere agli intervalli selezionati per ogni colonna.


st.write("---")

if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Categorical Inspection")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Transformation")
