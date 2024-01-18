import time

import numpy as np
import pandas as pd
import streamlit as st
import random
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
import os
import pandas_profiling
from pandas_profiling import *
import json


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
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

def save ():
    df = st.session_state['dataset']
    if st.session_state['Once']:
        with st.spinner("Please wait while the new dataset is profiled again.."):
            profileAgain(df)
    st.session_state['Once'] = False
    switch_page("Homepage")


profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
dfCol = st.session_state['dfCol']

st.title("Null values handling")
with st.expander("Single column perspective", expanded=True):
    st.write("In this page you're allowed to select a column in order to replace its null values. You can also decide to drop the entire column too.")
    column = selectbox("Choose a column", dfCol)
    #column = st.multiselect("Replace all the values with", dfCol)
    if column != None:
        nullNum = df[column].isna().sum()
        if nullNum > 0:
            percentageNull = nullNum/len(df.index)*100
            mask = df[column].isna()
            preview = df[mask]
            st.write("This column has ", nullNum, " null values (" + "%0.2f" %(percentageNull) + "%)")
            st.write(preview)
            st.write("If you want to proceed click next, otherwise you can either select another column or come back to the Homepage")
            if st.button("Next"):
                st.session_state['from'] = 1
                st.session_state['y'] = 0
                st.session_state['arg'] = df[column].copy(deep=False)
                switch_page("null_values")
        else:
            st.error("This column doesn't have any null values, select another column.")
with st.expander("Dataset perspective", expanded=True):
    if column == None:
        columnsNull = st.columns([1 for i in range(len(df.columns))], gap="small")
        for i in range(len(df.columns)):
            col = df.columns[i]
            nullPercentage = df.iloc[:, i].isna().sum() / len(df.index) * 100
            nullPercentageString = "{:.2f}".format(nullPercentage)
            with columnsNull[i]:
                st.write(f"**{col}**: {nullPercentageString}%")
        st.info("If you want to replace all the null values of the dataset in 2 click, select here with which value")
        col = st.columns([1,1,1.5])
        with col[0]:
            fillingOpStr = st.radio("Filling options for columns of type object:",("", "Following Value", "Previous Value","Mode", "Custom Value"), index=0,key=107)
        with col[1]:
            fillingOpNum = st.radio(f"Filling options for numeric columns :",("", "Min", "Max", "Avg", "0", "Mode"),index=0)
        if fillingOpNum != "" and fillingOpStr != "":
            copyPreview = df.copy()
            k = 100
            for dfCol in df.columns:
                k += 1
                if df[dfCol].isna().sum() > 0:
                    #st.write(df[dfCol].isna().sum(), dfCol)
                    #st.write(str(df[dfCol].dtype), dfCol)
                    if report["variables"][dfCol]["type"] != "Variable.S_TYPE_UNSUPPORTED":
                        if str(df[dfCol].dtype) == "object":
                            if fillingOpStr == "Following Value":
                                copyPreview[dfCol].fillna(method="bfill", inplace=True)
                            elif fillingOpStr == "Previous Values":
                                copyPreview[dfCol].fillna(method="ffill", inplace=True)
                            elif fillingOpStr == "Mode":
                                #st.write(copyPreview['DescrizioneVia'].isna().sum())
                                #st.write(dfCol, type(df[dfCol]))
                                #strMode = report["variables"][dfCol]["top"]
                                try:
                                    #st.write(copyPreview['DescrizioneVia'].isna().sum())
                                    strMode = report["variables"][dfCol]["top"]
                                    copyPreview[dfCol].fillna(strMode, inplace=True)
                                    st.write(strMode, dfCol)
                                except:
                                    st.write(copyPreview['DescrizioneVia'].isna().sum())
                                    st.error(f"For column **{dfCol}** is not possible to identify the mode value, no changes have been applied.")
                            elif fillingOpStr == "Custom Value":
                                customValue = st.text_input(f"Please insert the custom value you want to use for column **{dfCol}**:", key=k)   
                                if len(customValue) != 0:
                                    copyPreview[dfCol].fillna(customValue, inplace=True)
                        elif str(df[dfCol].dtype) == "float64" or str(df[dfCol].dtype) == "Int64":
                            if fillingOpNum == "Min":
                                minValue = report["variables"][dfCol]["min"]
                                copyPreview[dfCol].replace([np.nan], minValue, inplace=True)
                            elif fillingOpNum == "Max":
                                maxValue = report["variables"][dfCol]["max"]
                                copyPreview[dfCol].replace([np.nan], maxValue,inplace=True)
                            elif fillingOpNum == "Avg":
                                avgValue = report["variables"][dfCol]["mean"]
                                avgValue2 = round(avgValue)
                                copyPreview[dfCol].replace([np.nan], avgValue2,inplace=True)
                            elif fillingOpNum == "0":
                                copyPreview[dfCol].replace([np.nan], 0,inplace=True)
                            elif fillingOpNum == "Mode":
                                copyPreview[dfCol].fillna(copyPreview.mode()[0], inplace=True)
            nullBeforePercentage = report["table"]["p_cells_missing"]*100
            st.write("Missing values before the filling: ", report["table"]["n_cells_missing"], "(~", "%0.2f" %(nullBeforePercentage) + "%)")
            nullAfter = copyPreview.isna().sum().sum()
            if nullAfter > 0:
                nullAfterPercentage = nullAfter / df.size * 100
                st.write("Missing values after the filling: ", nullAfter, "(~", "%0.2f" %(nullAfterPercentage) + "%)")
            else:
                st.write("Missing values after the filling: ", nullAfter)
            st.write(copyPreview)
            st.session_state['dataset'] = copyPreview.copy()
            st.session_state['Once'] = True
            if st.button("Save"):
                df = st.session_state['dataset']
                if st.session_state['Once']:
                    with st.spinner("Please wait while the new dataset is profiled again.."):
                        profileAgain(df)
                st.session_state['Once'] = False
                switch_page("Homepage")

                #st.session_state['y'] = 
                #reload the page, profile again the dataset -> done
                        



if st.button("Back to Homepage"):
    switch_page("Homepage")

