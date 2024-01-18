import time

import numpy as np
import pandas as pd
import streamlit as st
import random
from streamlit_extras.switch_page_button import switch_page
import os
from pandas_profiling import *
import json

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

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

st.markdown(
    """ <style>
            div[role="tablist"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)


parentPage = st.session_state['from']   #0 dataset info, 1 cl by col

dfCol = st.session_state['arg'] #dfCol is a series here
dfCol1 = st.session_state['arg1']
profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
corr = st.session_state['correlation']

if st.session_state['y'] == 0:
    st.session_state['toBeProfiled'] = True
    stringTitle = "Data redundancy between " + dfCol1.name + " and " + dfCol.name
    st.title(stringTitle)
    col1_1, col2_1, col3_1, col4_1 = st.columns(4, gap='small')
    with col1_1:
        st.subheader("Preview of the 2 columns")
        st.write(df[[dfCol1.name, dfCol.name]].head(50))
    with col2_1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        #stringinfo = f"The column **{dfCol1.name}** and the column **{dfCol.name}** have a correlation of "+"%0.2f" %(corr) + "%"
        #st.info(stringinfo)
        st.write(f"The column **{dfCol1.name}** and the column **{dfCol.name}** have a correlation of ""%0.2f" %(corr) , "%")
    st.write("**Available actions**")
    strDrop = "Drop " + dfCol.name
    strDrop1 = "Drop " + dfCol1.name
    tab1, tab2, tab3, tab4 = st.tabs(["--", strDrop, strDrop1, "Do nothing"])

    with tab1: #default case for visualization's reasons, don't modify
        ()
    with tab2:
        dfCopy = df.copy()
        strWarning1 = f"Do you really want to drop the entire column **{dfCol.name}**?"
        st.subheader("New dataset preview")
        dfCopy = dfCopy.drop(dfCol.name, axis=1)
        st.write(dfCopy.head(20))
        st.warning(strWarning1)
        if st.button("Confirm", key=3):

            #TODO
            #update the dataframe
            #do the profile again
            #remove the old report
            #load the new one
            df = dfCopy.copy()
            successMessage = st.empty()
            successString = "Column successfully dropped! Please wait while the dataframe is profiled again.."
            successMessage.success(successString)
            if st.session_state['toBeProfiled'] == True:
                profileAgain(df)
            st.session_state['toBeProfiled'] = False
            successMessage.success("Profiling updated!")
            st.session_state['y'] = 1
            st.experimental_rerun()
    with tab3:
        dfCopy = df.copy()
        strWarning2 = f"Do you really want to drop the entire column **{dfCol1.name}**?"
        st.subheader("New dataset preview")
        dfCopy = dfCopy.drop(dfCol1.name, axis=1)
        st.write(dfCopy.head(20))
        st.warning(strWarning2)
        if st.button("Confirm", key=4):
            df = dfCopy.copy()
            successMessage = st.empty()
            successString = "Column successfully dropped! Please wait while the dataframe is profiled again.."
            successMessage.success(successString)
            if st.session_state['toBeProfiled'] == True:
                profileAgain(df)
            st.session_state['toBeProfiled'] = False
            st.session_state['y'] = 1
            successMessage.success("Profiling updated!")
            st.experimental_rerun()
    with tab4: #none
        ()
    st.markdown("""---""")
    if parentPage == 0:
        if st.button("Back to Dataset Info!"):
            switch_page("dataset_info")
    elif parentPage == 1:
        if st.button("Back to column by columns"):
            switch_page("col_by_col")
elif st.session_state['y'] == 1:
    if parentPage == 0:    
        st.success("Redirecting to dataset_info..")
        switch_page("dataset_info")
    elif parentPage == 1:
        st.success("Redirecting to column by column..")
        switch_page("col_by_col")
else:
    ()
