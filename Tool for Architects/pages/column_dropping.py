import json
import time
import functional_dependencies
import jellyfish
import numpy as np
import pandas as pd
import pandas_profiling
import streamlit as st
from pandas_profiling import *
from streamlit_extras.switch_page_button import switch_page
from streamlit_pandas_profiling import st_profile_report
from streamlit_extras.echo_expander import echo_expander
from streamlit_extras.toggle_switch import st_toggle_switch
from streamlit_extras.stoggle import stoggle
from random import *
import os
import streamlit_nested_layout
import webbrowser

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
    height: 3.5em;
    width:auto;
}
</style>""", unsafe_allow_html=True)

def update_multiselect_style():
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb="tag"] {
                height: fit-content;
            }
            .stMultiSelect [data-baseweb="tag"] span[title] {
                white-space: normal; max-width: 100%; overflow-wrap: anywhere;
            }
            span[data-baseweb="tag"] {
                background-color: indianred !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

update_multiselect_style()

df = st.session_state['df']
dfCol = st.session_state['dfCol']
profile = st.session_state['profile']
report = st.session_state['report']
avoid = st.session_state['avoid'] #0 from Homepage, 1 from col_by_col

st.title("Column Dropping")

if st.session_state['y'] == 0:
    dfPreview = df.copy()
    if avoid == 0:
        listCol = dfPreview.columns
        listCol = listCol.insert(0, "Don't drop anything")
        colToDrop = st.multiselect("Select one or more column to drop", listCol, "Don't drop anything")
        if (len(colToDrop) > 0) and (colToDrop.count("Don't drop anything") == 0):
            for col in colToDrop:
                dfPreview = dfPreview.drop(col, axis=1)
            st.subheader("Real time preview")
            st.write(dfPreview)
            if st.button("Save"):
                st.session_state['newdf'] = dfPreview.copy()
                st.session_state['y'] = 1
                st.session_state['toBeProfiled'] = True
                st.experimental_rerun()
        st.markdown("---")
        if st.button("Back To Homepage"):
            switch_page("homepage")
    elif avoid == 1:
        col = st.session_state['arg']
        dfPreview = dfPreview.drop(col, axis=1)
        st.subheader("Preview")
        st.write("Dropped column ", col)
        st.write(dfPreview)
        if st.button("Save"):
                st.session_state['newdf'] = dfPreview.copy()
                st.session_state['y'] = 1
                st.session_state['toBeProfiled'] = True
                st.experimental_rerun()
        st.markdown("---")
        if st.button("Back to column by column"):
            switch_page("col_by_col")




elif st.session_state['y'] == 1:
    successMessage = st.empty()
    successString = "Please wait while the dataframe is profiled again with all the applied changes.."
    df = st.session_state['newdf']
    if st.session_state['toBeProfiled'] == True:
        successMessage.success(successString)
        #st.markdown('''(#automatic)''', unsafe_allow_html=True)
        with st.spinner(" "):
            profileAgain(df)
    successMessage.success("Profiling updated! You can go to 'dataset info' in order to see the report of the new dataset or comeback to the homepage.")
    st.session_state['toBeProfiled'] = False
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,10], gap='small')
    with col1:
        if st.button("Back To Homepage"):
            switch_page("homepage")
    with col2:
        if st.session_state['avoid'] == 1:
            if st.button("Back to column by column"):
                switch_page("col_by_col")
#st.markdown("---")


