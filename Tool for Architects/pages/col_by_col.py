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
import streamlit.components.v1 as components

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

def read_html():
    with open("col_by_col.html") as f:
        return f.read()


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:7em;
    width:7em;
}
</style>""", unsafe_allow_html=True)

e = st.markdown("""
<style>
div[data-testid="stExpander"] div[role="button"] p {
    font-size: 2rem;
}
</style>""", unsafe_allow_html=True)

df = st.session_state['df']
dfCol = st.session_state['dfCol']
profile = st.session_state['profile']
report = st.session_state['report']
correlations = profile.description_set["correlations"]
phik_df = correlations["phi_k"]
count = st.session_state['counter']

st.title("Column by column")
st.subheader("In this page you'll visualize all the information of every column")

col1, col2, col3 = st.columns([2,10,2])
with col1:
    if count > 0:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        inner_prev_space = st.columns([1,3.5])
        with inner_prev_space[1]:
            if st.button("Previous"):
                st.session_state['counter'] = count - 1
                st.experimental_rerun()

with col2:
    flagDistinct = False
    flagNull = False
    flagSplit = False
    flagDrop = False
    col = df.columns[count]
    with st.expander(col, expanded=True):
        inner_cols = st.columns([2,0.3,5])
        with inner_cols[0]:
            st.dataframe(df.iloc[:, count], use_container_width=True)
        with inner_cols[2]:
            inner_inner_cols = st.columns([1,0.1,1])
            with inner_inner_cols[0]:
                st.subheader("Statistics")
                for i in range(0,10):
                    if df[col].dtype == "float64" or df[col].dtype == "Int64":
                        break
                    elif " " in str(df[col][i]) :
                        flagSplit = True
                        break
                nullNum = df[col].isna().sum()
                distinctNum = len(pd.unique(df[col]))
                testCount = df[col].drop_duplicates().size
                if report["variables"][dfCol[int(count)]]["type"] != "Variable.S_TYPE_UNSUPPORTED":
                    columnUnique = report["variables"][dfCol[int(count)]]["n_unique"]
                    percentageUnique = columnUnique/len(df.index)*100
                else:
                    columnUnique = "Not available"
                st.write("Column type is: ",str(df[col].dtype))
                st.write("Null values: ",nullNum)
                percentageNull = nullNum/len(df.index)*100
                st.write("Percentage of null values: ","%0.2f" %(percentageNull) + "%")
                flagNull = True
                if nullNum == 0:
                    flagNull = False
                if percentageNull > 25:
                    st.error("This attribute has more than 25" + "%" + " of null values")
                st.write("Distinct values: ", distinctNum)
                if distinctNum == 1:
                    st.warning("This column has the same value for all the rows")
                    flagDrop = True
                percentageDistinct = distinctNum/len(df.index)*100
                st.write("Percentage of distinct values: ","%0.2f" %(percentageDistinct) + "%")
                if percentageDistinct < 4 and distinctNum != 1:
                    flagDistinct = True
                    st.warning("This attribute is almost a category for the dataset")
                st.write("Unique values: ", columnUnique)
                if columnUnique != "Not available":
                    st.write("Percentage of unique values: ","%0.2f" %(percentageUnique) + "%")
                    if percentageUnique > 90:
                        st.warning("This attribute is a possible candidate for primary key!")
            with inner_inner_cols[2]:
                corrList = []
                #st.write(phik_df[col])
                if col in phik_df.columns:
                    st.subheader("Correlation")
                    for y in range(0, len(phik_df.index)):
                        if phik_df.columns[y] != col:
                            x = float(phik_df[col][y])*100
                            if x > 85:
                                st.write(f"Correlation with **{str(phik_df.columns[y])}**  is: ", "%0.2f" %(x) , "%")
                                st.write("")
                                corrList.append([phik_df.columns[y], x])
                    

with col3:
    if count < (len(df.columns) - 1):
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        inner_next_space = st.columns([1,3.5])
        with inner_next_space[1]:
            if st.button("Next"):
                st.session_state['counter'] = count + 1
                st.experimental_rerun()
x = 0
if flagNull == True:
    x = 1
st.write("")
st.write("")
finalCols = st.columns([1.5,1,1,1,1,1,1,1,1,1],gap="large")
with finalCols[x]:
    if flagNull == True:
        if st.button("Handle null values"):
                        st.session_state['from'] = 2
                        st.session_state['y'] = 0
                        st.session_state['arg'] = df.iloc[:, count].copy(deep=False)
                        switch_page("null_values")

if flagDrop == True:
    x += 1
    with finalCols[x]:
        if st.button("Drop column"):
            st.session_state['arg'] = col
            st.session_state['avoid'] = 1
            st.session_state['y'] = 0
            switch_page("column_dropping")
if flagDistinct == True:
    x += 1    
    with finalCols[x]:
        if st.button("Manage categories"):
            st.session_state['y'] = 0
            st.session_state['arg'] = df.iloc[:, count].copy(deep=False)
            switch_page("category")
if flagSplit == True:
    x += 1
    with finalCols[x]:
        if st.button("Split column"):
            st.session_state['colName'] = col
            st.session_state['y'] = 0
            st.session_state['avoid'] = 1
            switch_page("column_splitting")
if len(corrList) > 0:
    for item in corrList:
        x += 1
        with finalCols[x]:
            key = x + 50
            string =f"Correlation with **{str(item[0])}**"
            if st.button(string, key=key):
                st.session_state['y'] = 0
                st.session_state['from'] = 1
                st.session_state['correlation'] = item[1]
                st.session_state['arg'] = df.iloc[:, count].copy(deep=False)
                st.session_state['arg1'] = df[item[0]].copy(deep=False)
                switch_page("data_correlation")

st.markdown("---")
if st.button("Homepage"):
    switch_page("Homepage")

#components.html(
#    read_html(),
#    height=0,
#    width=0,
#)




