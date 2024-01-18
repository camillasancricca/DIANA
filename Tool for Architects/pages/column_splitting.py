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

def clean2 ():
    slate2.empty()
    st.session_state['y'] = 2
    st.session_state['toBeProfiled'] = True

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
avoid = st.session_state['avoid'] #0 from Homepage, 1 from col_by_col

st.title("Column splitting")
slate1 = st.empty()
body1 = slate1.container()

slate2 = st.empty()
body2 = slate2.container()

with body1:
    if st.session_state['y'] == 0:  #choose the values to replace
        if avoid == 0:
            st.subheader("Dataset preview")
            st.write(df)
            columns = list(df.columns)
            columnsFiltered = []
            for col in columns:
                for i in range(0,10):
                    if df[col].dtype == "float64" or df[col].dtype == "Int64":
                        break
                    #elif " " in df[col][i] :
                    else:
                        columnsFiltered.append(col)
                        break
            column = st.selectbox("Select the column to be splitted", columnsFiltered)
        elif avoid == 1:
            st.markdown("---")
            column = st.session_state['colName']
            string = f"Splitting column **{column}**"
            st.write(string)
        delimiters = [" ", ",", ";", " ' ", "{", "}", "[", "]", "(", ")", " \ ", "/", "-", "_", ".", "|"]
        if column != None:
            colPreview = df[column].copy()
            importantDelimiters = []
            importantCounters = []
            for item in delimiters:
                counter = 0
                for i in range(int(len(df.index))):  #search for a delimiter in the column
                    x = str(colPreview[i]).find(item)
                    if x > 0:
                        counter += 1
                if counter >= (len(df.index)/20):  #if is present at least in 5% of the lines then we should propose it!
                    importantDelimiters.append(item)
                    importantCounters.append(counter)
            
            for i in range(0, len(importantDelimiters)):
                if importantDelimiters[i] == " ":
                    importantDelimiters[i] = "First space appearance"
            importantDelimiters.insert(0,"None")
            importantCounters.insert(0, "None")
            delimiter = st.selectbox("Select the delimiter ", importantDelimiters)
            if delimiter != "None":
                num = importantCounters[(importantDelimiters.index(delimiter))]
                percentage = num / len(df.index) * 100
                infoString = f"This delimiter appears {num} times (" + str("%0.2f" %(percentage)) + "%) within this column"
                for i in range(len(df.index)):
                    if delimiter == "First space appearance":
                        if " " in str(colPreview[i]):
                            flag = 0
                            string = str(colPreview[i]).split(" ", 1)
                            break
                    else:
                        #infoString = f"This delimiter appears {importantCounters[(importantDelimiters.index(delimiter))]} times within this column"
                        if delimiter in str(colPreview[i]):
                            flag = 1
                            string = str(colPreview[i]).split(delimiter, 1)
                            break
                st.info(infoString)
                if len(string) == 1:
                    st.error("It's not possible to split this column")
                
                else:
                    if len(string[0]) == 0:
                        string[0] = None
                    if len(string[1]) == 0:
                        string[1] = None
                    data = [string]
                    st.write(f"One-line preview of the first splittable row, **you're strongly recommended to do not apply splitting to numeric columns**")
                    oneLinePreview = pd.DataFrame(data, columns=['First column', 'Second column'])
                    st.write(oneLinePreview)
                    firstColumn = st.text_input("Insert the name of the new 'First column'")
                    secondColumn = st.text_input(f"Insert the name of the new 'Second column' **and press Enter**")
                    columnsLower = [str(element).lower() for element in df.columns]
                    firstColumnLower = firstColumn.lower()
                    secondColumnLower = secondColumn.lower()
                    if firstColumn == "" or secondColumn == "":
                        ()
                    elif firstColumnLower == secondColumnLower:
                        st.error("The columns name should be distinct")
                    elif firstColumnLower in columnsLower or secondColumnLower in columnsLower:
                        st.error("One of the two columns is already present in the dataset")
                    else:
                        if num < len(df.index):
                            st.write("")
                            str1 =f"Fill only  {firstColumn}  and None in {secondColumn}"
                            str2 =f"Fill only  {secondColumn}  and None in {firstColumn}"
                            choice = st.radio("With the rows that do not contain the delimiter and consequentially can't be split, what would you like to do?",["None", str1, str2])
                            
                            if choice == str1:
                                st.session_state['itemToPass'] = [column, firstColumn, secondColumn, delimiter, 1]
                                if st.button("Go!", on_click=clean1):
                                    ()
                            elif choice == str2:
                                st.session_state['itemToPass'] = [column, firstColumn, secondColumn, delimiter, 2]
                                if st.button("Go!", on_click=clean1):
                                    ()
                        else:
                            st.session_state['itemToPass'] = [column, firstColumn, secondColumn, delimiter, 1]
                            if st.button("Go!", on_click=clean1):
                                ()
        st.markdown("---")
        if avoid == 0:
            if st.button("Back to Homepage"):
                switch_page("Homepage")
        elif avoid == 1:
            if st.button("Back to Column by column"):
                switch_page("col_by_col")
with body2:
    if st.session_state['y'] == 1:
        dfPreview = st.session_state['df'].copy()
        item = st.session_state['itemToPass']
        #st.write(item)
        if item[3] == "First space appearance":
            item[3] = " "
        columns = list(dfPreview.columns)
        colIndex = columns.index(item[0])
        valFirst = []
        valSecond = []
        for i in range (0, len(df.index)):
            if item[3] in str(dfPreview[item[0]][i]):
                splitted = str(dfPreview[item[0]][i]).split(item[3], 1)
                valFirst.append(splitted[0])
                if len(splitted[1]) == 0:
                    valSecond.append(None)
                else:
                    valSecond.append(splitted[1])
            elif item[4] == 1: #copy in the first col and null in the second column
                valFirst.append(dfPreview[item[0]][i])
                valSecond.append(None)
            elif item[4] == 2: #copy in the second column
                valFirst.append(None)
                valSecond.append(dfPreview[item[0]][i])
        #ser = pd.Series(list)
        #st.write(valSecond)
        dfPreview.insert(loc=colIndex, column = str(item[1]), value=valFirst)
        dfPreview.insert(loc=(colIndex+1), column = str(item[2]), value=valSecond)
        dfPreview.drop(item[0], inplace=True, axis=1)
        st.write(dfPreview)
        st.warning("This action will be permanent")
        col1, col2, col3 = st.columns([1,1,10], gap='small')
        st.session_state['newdf'] = dfPreview.copy()
        with col1:
            if st.button("Save!", on_click=clean2):
                ()
                #st.session_state['y'] = 2
                #st.session_state['toBeProfiled'] = True
                #st.experimental_rerun()
        with col2:
            if st.button("Back"):
                st.session_state['Once'] = True
                st.session_state['y'] = 0
                #st.session_state['toBeProfiled'] = True     
                st.experimental_rerun()
        st.markdown("---")
        if avoid == 0:
            if st.button("Homepage"):
                switch_page("Homepage")
        elif avoid == 1:
            if st.button("Column by column"):
                switch_page("col_by_col")

if st.session_state['y'] == 2:
    if st.session_state['toBeProfiled'] == True:
        df = st.session_state['newdf']
        successMessage = st.empty()
        successString = "Please wait while the dataframe is profiled again with all the applied changes.."
        successMessage.success(successString)
        with st.spinner(" "):
            profileAgain(df)
        successMessage.success("Profiling updated!")
        st.session_state['toBeProfiled'] = False
        st.subheader("New dataset")
        st.write(df)
    st.markdown("---")
    if avoid == 0:
            if st.button("Homepage"):
                switch_page("Homepage")
    elif avoid == 1:
        if st.button("Column by column"):
            switch_page("col_by_col")



