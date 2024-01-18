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


def profileAgain(df):
    if os.path.exists("newProfile.json"):
        os.remove("newProfile.json")
    profile = ProfileReport(df)
    profile.to_file("newProfile.json")
    with open("newProfile.json", 'r') as f:
        report = json.load(f)
    st.session_state['profile'] = profile
    st.session_state['report'] = report
    st.session_state['df'] = df
    newColumns = []
    for item in df.columns:
        newColumns.append(item)
    st.session_state['dfCol'] = newColumns




m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:4em;
    width:auto;
}
</style>""", unsafe_allow_html=True)

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
dfCol = st.session_state['dfCol']

st.title("View the entire dataset")
st.info(f"**Interact with the table!** Click a column header to sort its values or use ctrl+F to search something within the table. Also column width is manually resizable.")
st.dataframe(df)

st.write("Number of rows: ", report["table"]["n"])
st.write("Number of columns:", report["table"]["n_var"])
x = report["table"]["p_cells_missing"]*100
st.write("Total number of missing values: ", report["table"]["n_cells_missing"], "(~", "%0.2f" %(x) + "%)")
if st.button("Homepage"):
    switch_page("Homepage")

