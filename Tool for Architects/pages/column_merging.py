import time
import numpy as np
import pandas as pd
import streamlit as st
import random
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
from streamlit_extras.st_keyup import st_keyup
import os
from pandas_profiling import *
import json
import numpy as np

def update_selectbox_style():
    st.markdown(
        """
        <style>
            .stSelectbox [data-baseweb="select"] div[aria-selected="true"] {
                white-space: normal; overflow-wrap: anywhere;
                :first-child{
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
update_selectbox_style()

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:4em;
    width:auto;
}
</style>""", unsafe_allow_html=True)

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

def clean1 ():
    slate1.empty()
    st.session_state['y'] = 1
    st.session_state['toBeProfiled'] = True
def clean2 ():
    slate2.empty()
    st.session_state['y'] = 2
    

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']

st.title("Columns merging")
slate1 = st.empty()
body1 = slate1.container()

slate2 = st.empty()
body2 = slate2.container()
name = ""
finalName = ""
with body1:
    if st.session_state['y'] == 0:  #choose the values to replace
        
        st.subheader("Dataset preview")
        st.write(df.head(200))
        columns = list(df.columns)
        columns.insert(0, "Select a column")
        columnsCopy = columns.copy()
        column1 = st.selectbox("Select the first column to be merged", columns,key=1)
        if column1 != "Select a column":
            columnsCopy.remove(column1)
            column2 = st.selectbox("Select the second column to be merged", columnsCopy,key=2)
            if column2 != "Select a column":
                    name = st.selectbox("Select the name of the new column: a custom one or the name of one of the previous columns", [" ", "Custom name", column1, column2])
                    if name == "Custom name":
                        finalName = st.text_input("Type the new name and press enter")
                    elif name == column1 or name == column2:
                        finalName = name
                    if finalName != "":
                        st.subheader("Preview of the new dataset")
                        columns = list(df.columns)
                        colIndex = columns.index(column1)
                        if colIndex > 0:
                            colIndex -= 1
                        dfPreview = df.copy()
                        dfColPreview = pd.Series()
                        #IF NOT NULL
                        dfColPreview = dfPreview[column1].astype(str) + " " + dfPreview[column2].astype(str)
                        dfPreview.drop(column1, inplace=True, axis=1)
                        dfPreview.drop(column2, inplace=True, axis=1)
                        dfPreview.insert(loc=colIndex, column = finalName, value = dfColPreview)
                        st.write(dfPreview.head(50))
                        st.warning("This action will be permanent")
                        st.session_state['dfPreview'] = dfPreview.copy()
                        if st.button("Save", on_click=clean1):
                            ()                   
        st.markdown("---")
        if st.button("Back to Homepage"):
            switch_page("Homepage")

if st.session_state['y'] == 1:
    if st.session_state['toBeProfiled'] == True:
        df = st.session_state['dfPreview']
        successMessage = st.empty()
        successString = "Please wait while the dataframe is profiled again with all the applied changes.."
        successMessage.success(successString)
        with st.spinner(" "):
            profileAgain(df)
        successMessage.success("Profiling updated!")
        st.session_state['toBeProfiled'] = False
        st.subheader("New dataset")
        st.write(df.head(50))
    st.markdown("---")
    if st.button("Homepage"):
        switch_page("Homepage")



