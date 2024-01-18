from cgitb import small
import os
import webbrowser
import streamlit as st
import time
import pandas as pd
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
import streamlit_nested_layout

st.set_page_config(page_title="Homepage", layout="wide", initial_sidebar_state="collapsed")

#def app():

df = st.session_state['df']
#st.write(df.head())

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
    font-size: 2rem;
}
</style>""", unsafe_allow_html=True)

st.title("Homepage")
#st.markdown("- 💱 **Renaming**: you can rename one or more columns")
#st.info("You can operate one or more wrangling operations together. Remember to let the tool know (from the sidebar) that you finished your operations in order to generate the download .csv button.")
#st.error("⚠️REMEMBER: when you are one the wrangling page do NOT call back the data profiling page. You will lost every operation done!")
st.write(" ")
st.write(" ")
pageCol = st.columns([1,1,1])
with pageCol[0]:
    with st.expander(f"**Data profiling**", expanded=True):
        st.write("")
        profilingRow1 = st.columns([1,1,1])
        st.write("")
        profilingRow2 = st.columns([1,1,1])
        with profilingRow1[0]:
            if(st.button("Profiling", key=1)):
                st.session_state['Once'] = True
                switch_page("profiling")
        with profilingRow1[1]:
            if st.button("Dataset Info", key=2):
                st.session_state['status'] = 0
                switch_page("dataset_info")
        with profilingRow1[2]:
            if st.button("Show Entire dataset", key=3):
                switch_page("visualize_dataset")
        with profilingRow2[0]:
            if st.button("Download dataset", key=4):
                switch_page("download_dataset")
        with profilingRow2[1]:
            if st.button("Info by column", key=12):
                st.session_state['counter'] = 0
                switch_page("col_by_col")
with pageCol[1]:
    with st.expander(f"**Data cleaning**", expanded=True):
        st.write("")
        cleaningRow1 = st.columns([1,1,1])
        st.write("")
        cleaningRow2 = st.columns([1,1,1])
        with cleaningRow1[0]:
            if st.button("Duplicate detection", key=6):
                st.session_state['y'] = 0
                switch_page("duplicate_detection")
        with cleaningRow1[1]:
            if st.button("Null values", key=9):
                switch_page("null_values_selection")
        with cleaningRow1[2]:
            if st.button("Column dropping", key=14):
                st.session_state['avoid'] = 0
                st.session_state['y'] = 0
                switch_page("column_dropping")
            
        with cleaningRow2[0]:
            if(st.button("Automatic Cleaning", key=7)):
                st.session_state['y'] = 0
                st.session_state['widget'] = 500
                switch_page("automatic")
        with cleaningRow2[1]:
            if st.button("Suggested actions", key=5):
                st.session_state['Once'] = True
                st.session_state['y'] = 0
                switch_page("suggested_actions")
with pageCol[2]:
    with st.expander(f"**Data wrangling**", expanded=True):
        st.write("")
        wranglingRow1 = st.columns([1,1,1])
        st.write("")
        wranglingRow2 = st.columns([1,1,1])
        with wranglingRow1[0]:
            if st.button("Values editing", key=8):
                st.session_state['y'] = 0
                switch_page("value_filtering")
        with wranglingRow1[1]:
            if st.button("Column renaming", key=10):
                st.session_state['y'] = 0
                switch_page("column_renaming")
        with wranglingRow2[0]:
            if st.button("Column splitting", key=11):
                st.session_state['avoid'] = 0
                st.session_state['y'] = 0
                switch_page("column_splitting")
        with wranglingRow2[1]:
            if st.button("Columns merging", key=13):
                st.session_state['y'] = 0
                switch_page("column_merging")
st.subheader("Dataset")
st.write(df)
st.write("---")
if button("Change dataset", key="changedataset"):
    st.warning("If you don't have downloaded the modified dataset you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")