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

e = st.markdown("""
<style>
div[data-testid="stExpander"] div[role="button"] p {
    font-size: 1rem;
}
</style>""", unsafe_allow_html=True)

df = st.session_state['df']
dfCol = st.session_state['dfCol']
profile = st.session_state['profile']
report = st.session_state['report']
correlations = profile.description_set["correlations"]
phik_df = correlations["phi_k"]

st.title("Suggested actions")
slate1 = st.empty() 
body1 = slate1.container()

slate2 = st.empty()
body2 = slate2.container()

def clean1 ():
    slate1.empty()
    st.session_state['Once'] = True
    st.session_state['y'] = 1
    #st.session_state['2'] = True
    #st.session_state['toBeProfiled'] = True
    #st.experimental_rerun()

def clean2 ():
    slate2.empty()
    st.session_state['y'] = 2
    st.session_state['toBeProfiled'] = True
    #st.experimental_rerun()

def element_not_in_tuples(element, tuple_list):
    for tup in tuple_list:
        if element not in tup:
            return True
    return False

droppedList = []

ind = 1
correlationList = []
for col in phik_df.columns:
    if ind < (len(phik_df.columns) - 1):
        for y in range(ind, len(phik_df.columns)):
            x = float(phik_df[col][y])*100
            if x > 85:
                correlationList.append([col, str(phik_df.columns[y]), x])
        ind += 1

correlationSum = {}
for y in range(0, len(phik_df.columns)):
    x = 0
    z = 0
    for column in phik_df.columns:
        z = float(phik_df[column][y])
        x += z 
    correlationSum.update({str(phik_df.columns[y]) : x})


with body1:
    if st.session_state['y'] == 0:
        with st.expander("Dataset preview", expanded=True):
            st.write(df.head(100))
        with st.expander("Incomplete Rows", expanded=True):
            colNum = len(df.columns)
            threshold = round(0.4 * colNum) #if a value has 40% of the attribute = NaN it's available for dropping
            nullList = df.isnull().sum(axis=1).tolist()
            nullToDelete = []
            dfToDelete = df.copy()
            rangeList = list(range(len(nullList)))
            for i in range(len(nullList)):
                if nullList[i] >= threshold:
                    nullToDelete.append(i)
            if len(nullToDelete) > 0:
                notNullList = [i for i in rangeList if i not in nullToDelete]
                dfToDelete.drop(notNullList, axis=0, inplace=True)
                percentageNullRows = len(nullToDelete) / len(df.index) * 100
                infoStr = "This dataset has " + str(len(nullToDelete)) + " rows (" + str("%0.2f" %(percentageNullRows)) + "%) that have at least " + str(threshold) + " null values out of " + str(len(df.columns))
                st.info(infoStr)
                with st.expander("Expand to see all the incomplete rows"):
                    st.dataframe(dfToDelete)
                if st.checkbox("Drop these rows from the dataset", key=-1):
                    droppedList.append(["rows", nullToDelete])
            else:
                numm = colNum - threshold 
                successString = "The dataset has all the rows with at least " + str(numm) + " not null values out of " + str(len(df.columns))
                st.success(successString)
        with st.expander("Correlated columns", expanded=True):
            if len(correlationList) > 0:
                for i in range(0, len(correlationList)):
                    if correlationList[i][0] in df.columns and correlationList[i][1] in df.columns: 
                        if correlationSum[correlationList[i][0]] > correlationSum[correlationList[i][1]]:
                            x = 0
                            y = 1
                        else:
                            x = 1
                            y = 0
                        corr = float(phik_df[correlationList[i][0]][correlationList[i][1]])*100
                        strDropCorr =f"Columns **{correlationList[i][0]}** and  **{correlationList[i][1]}** are highly correlated (" + "%0.2f" %(corr) + f"%). You're suggested to drop **{correlationList[i][x]}** given the other correlation parameters between all the columns"
                        st.info(strDropCorr)
                        choice = st.radio("Select below the column to be dropped", ["None", correlationList[i][0], correlationList[i][1]], index=0)
                        if choice != "None":    
                            droppedList.append(["column", choice])
                        st.markdown("---")
            else:
                st.success("All the columns have a correlation parameters between them very low. This means they're all useful and descriptive for the dataset!")

        for col in df.columns:
            if len(pd.unique(df[col])) == 1:
                with st.expander("Useless columns", expanded=True):
                    strDropUnique = f"The column **{col}** has the same value for all the rows (that is '{df[col][1]}'). You're suggested to drop it because it doesn't add any information to the dataset"
                    st.info(strDropUnique)
                    key1 = hash(strDropUnique)
                    choice1 = st.radio("Select below", ["Keep the column", "Drop the column"], index=0, key=key1)
                    if choice1 != "Keep the column":
                        droppedList.append(["column", col])

        for col in df.columns:
            choiceNum = ""
            choiceObj = ""
            nullNum = df[col].isna().sum()
            percentageNull = nullNum/len(df.index)*100
            if percentageNull > 25:
                with st.expander("Null values", expanded=True):
                    if df[col].dtype == "object":
                        strObjFill = f"The column **{col}** has " + str("%0.2f" %(percentageNull)) + f"% of null values. Given it's an object column, you're suggested to replace them with its mode value, that is '{df[col].mode()}', or to directly drop it."
                        st.info(strObjFill)
                        choiceObj = st.radio("Select below", ["Don't replace null values", "Replace the null values with the mode", "Drop the column"])
                        if choiceObj != "Don't replace null values":
                            if choiceObj == "Replace the null values with the mode":
                                droppedList.append(["nullReplacedObj", col])
                            elif choiceObj == "Drop the column":
                                droppedList.append(["column", col])
                        #st.markdown("---")
                    elif df[col].dtype == "float64" or df[col].dtype == "Int64":
                        avgValue = "{:.2f}".format(report["variables"][col]["mean"])
                        strNumFill = f"The column **{col}** has " + str("%0.2f" %(percentageNull)) + f"% of null values. Given it's a numeric column, you're suggested to replace them with its mean value, that is '{round(float(avgValue))}', or directly drop it."
                        st.info(strNumFill)
                        choiceNum = st.radio("Select below", ["Don't replace null values", "Replace null values with the mean", "Drop the column"])
                        if choiceNum != "Don't replace null values":
                            if choiceNum == "Replace null values with the mean":
                                droppedList.append(["nullReplacedNum", col])
                            elif choiceNum == "Drop the column":
                                droppedList.append(["column", col])
                        #st.markdown("---")
                    else:
                        ()
                        #st.error("Unrecognized col. type") 
        if st.session_state['Once'] == True:
            with st.spinner("Checking redundancies in the data"):
                length = round(len(df.index)/10)
                limit = round(length * 60 / 100)
                st.session_state['redundancyList'] = []
                for col in df.columns:
                    for col1 in df.columns:
                        if col != col1 and col != "MUNICIPIO" and col1 != "MUNICIPIO":
                            dup = 0
                            for i in range(length):
                                if str(df[col][i]) in str(df[col1][i]):  #col1/arg1 ubicazione, col/arg descrizioneVia
                                    dup += 1
                            if dup > limit:
                                st.session_state['redundancyList'].append([col, col1])
            st.session_state['Once'] = False
        redundancyList = st.session_state['redundancyList']
        if len(redundancyList) > 0:
            with st.expander("Redundancies in the data", expanded=True):
                for item in redundancyList:
                    string = f"Some redundancy of information have been detected between column **{item[0]}** and **{item[1]}**, here a small preview of the two columns"
                    st.info(string)
                    st.write(df[[item[0], item[1]]].head(20))
                    choiceRed = st.radio("Select below", ["None", f"Drop column {item[0]}", f"Drop column {item[1]}", f"Remove only the duplicate information from column {item[1]}"], index=0)
                    if choiceRed != None:
                        if choiceRed == f"Drop column {item[0]}":
                            droppedList.append(["column", item[0]])
                        elif choiceRed == f"Drop column {item[1]}":
                            droppedList.append(["column", item[1]])
                        elif choiceRed == f"Remove only the duplicate information from column {item[1]}":
                            droppedList.append(["Redundancy", [item[0], item[1]]]) #should remove item[1][0] from item[1][1]
                    st.markdown("---")

        else:
            st.success("All the column have an high indepence between each other")



        st.session_state['droppedList'] = droppedList #update the array before doing the action    
        if len(droppedList) > 0:
            st.warning("After you apply the changes selected, a preview of the new dataset will be displayed with a recap of the actions performed")
            if st.button("Go!", on_click=clean1):
                ()
        st.markdown("---")
        if st.button("Homepage"):
            switch_page("Homepage")
with body2:
    if st.session_state['y'] == 1:
        
        #here we should perform all the actions in "droppedList" and profile again the dataset if the user wants
        
        dfPreview = df.copy()
        if st.session_state['Once'] == True:
            with st.spinner("Applying the required changes"):
                #st.write(st.session_state['droppedList'])
                for item in st.session_state['droppedList']:
                    if item[0] == "rows":
                        dfPreview.drop(item[1], axis=0, inplace=True)
                    elif item[0] == "column" and item[1] in dfPreview.columns:
                        #if str(item[1]) in dfPreview:
                        dfPreview.drop(item[1], axis=1, inplace=True)

                    elif item[0] == "nullReplacedNum":
                        col = item[1]
                        avgValue = "{:.2f}".format(report["variables"][col]["mean"])
                        dfPreview[col].fillna(round(float(avgValue)), inplace=True)
                    elif item[0] == "nullReplacedObj":
                        col = item[1]
                        strMode = report["variables"][col]["top"]
                        dfPreview[col].fillna(strMode, inplace=True)
                    elif item[0] == "Redundancy":
                        for i in range(len(dfPreview.index)):
                                if str(dfPreview[item[1][0]][i]) in str(dfPreview[item[1][1]][i]):
                                    dfPreview[item[1][1]][i] = str(dfPreview[item[1][1]][i]).replace(str(dfPreview[item[1][0]][i]), "")
            st.session_state['Once'] = False




        with st.expander("List of the actions performed", expanded=False):
            for item in st.session_state['droppedList']:
                if item[0] == "rows":
                    string = f"-  Dropped {len(item[1])} rows because they had more than 40% of null attributes."
                    st.write(string)
                elif item[0] == "column":
                    string = f"-  Dropped column **{item[1]}**."
                    st.write(string)
                elif item[0] == "nullReplacedNum":
                    avgValue = "{:.2f}".format(report["variables"][item[1]]["mean"])
                    string = f"-  Replaced all the null values present in column **{item[1]}** with its mean value ({round(float(avgValue))})."
                    st.write(string)
                elif item[0] == "nullReplacedObj": 
                    string = f"-  Replaced all the null values present in column **{item[1]}** with its mode value ({dfPreview[item[1]].mode()})."
                    st.write(string)
                elif item[0] == "Redundancy":
                    string = f"-  Removed from column **{item[1][1]}** the information that was present also in column **{item[1][0]}**"
                    st.write(string)
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
        if st.button("Homepage"):
            switch_page("Homepage")
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
        st.write(df.head(50))
    st.markdown("---")
    if st.button("Homepage"):
        switch_page("Homepage")
