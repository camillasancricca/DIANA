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

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
dfCol = st.session_state['dfCol']
oldfilename = st.session_state['filename']

st.title("Download Dataset")
@st.experimental_memo
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.write("The new dataset will be download in csv format, you can also change the name.")
st.warning("Don't include the .csv extension! It will be added automatically")
st.markdown("---")
list = oldfilename.split(".")
string = list[0] + "NEW"
name = st_keyup("New filename", string)
if name != "":
    name = name + ".csv"
    csv = convert_df(df)
    st.download_button(
        label="Download",
        data=csv,
        file_name=name,
        mime='text/csv',
    )
else:
    st.error("Specify a filename")
st.markdown("---")
if st.button("Back to Homepage"):
    switch_page("Homepage")

