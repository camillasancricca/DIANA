import time
import numpy as np
import pandas as pd
import streamlit as st
import random
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.st_keyup import st_keyup
import os
import json
from pandas_profiling import *
from streamlit_extras.stateful_button import button

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

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
    color: #ffffff; 
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

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
dfCol = st.session_state['dfCol']
oldfilename = st.session_state['filename']


st.title("Download Dataset")
def convert_df(dataframe):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return dataframe.to_csv().encode('utf-8')

st.markdown('<h3 style="color: white; font-size: 20px;">The new dataset will be download in csv format, you can also change the name.</h3>',unsafe_allow_html=True)
st.warning("Don't include the .csv extension! It will be added automatically")
st.markdown("---")
list = oldfilename.split(".")
string = list[0] + "NEW"

st.markdown('<h3 style="color: white; font-size: 20px;">New filename</h3>',unsafe_allow_html=True)
name = st_keyup("", string)
st.write(st.session_state.my_dataframe)
if name != "":
    name = name + ".csv"
    csv = convert_df(st.session_state.my_dataframe)
    st.download_button(
        label="Download",
        data=csv,
        file_name=name,
        mime='text/csv',
    )
else:
    st.error("Specify a filename")


st.markdown("---")
st.write("---")
if button("Change dataset", key="changedataset"):
    st.warning("If you don't have downloaded the modified dataset you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")