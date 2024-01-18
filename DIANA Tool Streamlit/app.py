from sqlite3 import connect

import pandas as pd
import streamlit as st
from PIL import Image
import requests



from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="Tool name", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>h1{color: black; font-family: 'Roboto', sans-serif;}</style>", unsafe_allow_html=True)

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



st.title("Welcome to the introduction page")

start_button = st.button("Let's start the analysis")
st.session_state['x'] = 0
if start_button:
    switch_page("upload")

