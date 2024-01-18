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
#st.markdown("<p><a id=scroll-to-bottom>Scroll to bottom</a></p>", unsafe_allow_html=True)
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

st.title("Automatic")
slate = st.empty()
body = slate.container()

def clean2 ():
    slate.empty()
    st.session_state['y'] = 2
    st.session_state['toBeProfiled'] = True
    #st.experimental_rerun()

def clean3 ():
    slate.empty()
    st.session_state['y'] = 3
    st.session_state['toBeProfiled'] = True
    #st.experimental_rerun()


NoDupKey = st.session_state['widget']

correlations = profile.description_set["correlations"]
phik_df = correlations["phi_k"]

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
    percentageNullRows = len(nullToDelete) / len(df.index) * 100

droppedList = []

#st.title("Automatic")
#st.subheader("Original dataset preview")
#st.dataframe(df.head(50))
#st.markdown("---")
with body:
    if st.session_state['y'] == 0:
        st.subheader("Original dataset preview")
        st.dataframe(df.head(50))
        st.markdown("---")
        st.write("Click the button to perform automatically all the actions that the system finds suitable for your dataset, later you'll have the possibility to check the preview of the new dataset and to rollback action by action.")
        if st.button("Go!"):
            st.session_state['y'] = 1
            st.session_state['once'] = True
            st.experimental_rerun()
        st.markdown("---")
        if st.button("Back To Homepage"):
            switch_page("homepage")

    elif st.session_state['y'] == 1:
        box = st.empty()
        dfAutomatic = df.copy()
        st.subheader("Original dataset preview")
        st.dataframe(df.head(50))
        st.markdown("---")
        if len(nullToDelete) > 0:
            stringDropAutomaticLoad = "Dropping the " + str(len(nullToDelete)) + " rows (" + str("%0.2f" %(percentageNullRows)) + "%) that have at least " + str(threshold) + " null values out of " + str(len(df.columns))
            stringDropRollback = "Check to rollback the drop of " + str(len(nullToDelete)) + " incomplete rows"
            stringDropAutomaticConfirmed = f"Successfully dropped **{str(len(nullToDelete))}** **rows** (" + str("%0.2f" %(percentageNullRows)) + "%) that had at least " + str(threshold) + " null values out of " + str(len(df.columns))
            #dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
            droppedList.append(["rows", nullToDelete])
            if st.session_state['once'] == True:
                with st.spinner(text=stringDropAutomaticLoad):
                    time.sleep(0.5)
            st.success(stringDropAutomaticConfirmed)
            with st.expander("Why I did it?"):
                st.write("Incomplete rows are one of the principal sources of poor information. Even by applying the imputing technique within these rows would just be almost the same as incresing the dataset's size with non-real samples.")
                if st.checkbox(stringDropRollback, value=False, key=len(nullToDelete)) == True:
                    droppedList = droppedList[ : -1]
                else:
                    dfAutomatic.drop(nullToDelete, axis=0, inplace=True)
        else:
            st.success("All the rows are complete at least for the 60%!")
        st.markdown("---")
        for i in range(0, len(correlationList)):
            if correlationList[i][0] in dfAutomatic.columns and correlationList[i][1] in dfAutomatic.columns: 
                if correlationSum[correlationList[i][0]] > correlationSum[correlationList[i][1]]:
                    x = 0
                    y = 1
                else:
                    x = 1
                    y = 0
                strDropAutomaticCorrLoad = "Dropping column " + correlationList[i][x] + " because of it's high correlation with column " + correlationList[i][y]
                strDropAutomaticCorrConfirmed = f"Successfully dropped column **{correlationList[i][x]}** because of its high correlation with column {correlationList[i][y]}"
                strDropCorrRollback = f"Check to rollback the drop of column **{correlationList[i][x]}**"
                #dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                droppedList.append(["column", correlationList[i][x]])
                if st.session_state['once'] == True:
                    with st.spinner(text=strDropAutomaticCorrLoad):
                        time.sleep(0.5)
                st.success(strDropAutomaticCorrConfirmed)
                with st.expander("Why I did it?"):
                    st.write("When two columns has an high correlation between each other, this means that the 2 of them together have almost the same amount of information with respect to have only one of them. ANyway some columns can be useful, for example, to perform aggregate queries. If you think it's the case with this column you should better rollback this action and keep it!")
                    if st.checkbox(strDropCorrRollback, key=NoDupKey) == True:
                        droppedList = droppedList[ : -1]
                    else:
                        dfAutomatic = dfAutomatic.drop(correlationList[i][x], axis=1)
                    NoDupKey = NoDupKey - 1
                #st.markdown("<p id=page-bottom>You have reached the bottom of this page!!</p>", unsafe_allow_html=True)
                st.markdown("---")
        for col in dfAutomatic.columns:
            #k = randint(1,100)
            if len(pd.unique(dfAutomatic[col])) == 1:
                strDropAutomaticDistLoad = "Dropping column " + col + " because has the same value for all the rows, that is " + str(dfAutomatic[col][1])
                strDropAutomaticDistConfirmed = f"Successfully dropped column **{col}** because has the same value for all the rows, that is {dfAutomatic[col][1]}"
                strDropDistRollback = f"Check to rollback the drop of column **{col}**"
                droppedList.append(["column", col])
                if st.session_state['once'] == True:
                    with st.spinner(text=strDropAutomaticDistLoad):
                        time.sleep(0.5)
                st.success(strDropAutomaticDistConfirmed)
                with st.expander("Why I did it?"):
                    st.write("The fact that all the rows of the dataset had the same value for this attribute, doesn't bring any additional information with respect to removing the attribute. A dumb example could be: imagine a table of people with name, surname, date of birth...Does make sense to add a column called 'IsPerson'? No, because the answer would be the same for all the rows, we already know that every row here represent a person.")
                    if st.checkbox(strDropDistRollback, key=100) == True:
                        droppedList = droppedList[ : -1]
                    else:
                        dfAutomatic = dfAutomatic.drop(col, axis=1)
                st.markdown("---")
        for col in dfAutomatic.columns:
            nullNum = dfAutomatic[col].isna().sum()
            distinct = dfAutomatic[col].nunique()
            percentageNull = nullNum/len(df.index)*100
            if percentageNull > 1:
                if dfAutomatic[col].dtype == "object":  #automatically fill with the mode
                    x = 0
                elif dfAutomatic[col].dtype == "float64" or dfAutomatic[col].dtype == "Int64":  #automatically fill with the average
                    x = 1
                else:
                    x = 2
                    st.error("Unrecognized col. type")
                if x != 2:
                    strFillAutomaticLoad = "Replacing all the " + str(nullNum) + " (" + "%0.2f" %(percentageNull) + "%) null values of column " + col
                    strFillAutomaticRollback = f"Check to rollback the replacement of all the null values in column **{col}**"
                    originalCol = dfAutomatic[col].copy(deep=False)
                if x == 0:
                    try:
                        strMode = report["variables"][col]["top"]
                        dfAutomatic[col].fillna(strMode, inplace=True)
                        strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str("%0.2f" %(percentageNull)) + f"%) null values of the column **{col}** with the mode: {strMode}"
                        explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible this attribute, the mode is the value less invasive in terms of filling.  In the null values you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                    except:
                        ()                
                elif x == 1:
                    avgValue = "{:.2f}".format(report["variables"][col]["mean"])
                    dfAutomatic[col].fillna(round(round(float(avgValue))), inplace=True)
                    strFillAutomaticConfirmed = f"Successfully replaced all the {nullNum} (" + str("%0.2f" %(percentageNull)) + f"%) null values of the column **{col}** with the average value: {round(float(avgValue))}"
                    explanationWhy = "Unfortunately the column had a lot of null values. In order to influence less as possible the average value of this attribute, the mean is one of the best solution for the replacement. In the null values page you'll have the possibility also to choose other values. If you want so, remind to rollback this change in order to still have the null values in your dataset."
                if x == 0 or x == 1:
                    droppedList.append(["nullReplaced", col, dfAutomatic[col]])
                    if st.session_state['once'] == True:
                        with st.spinner(text=strFillAutomaticLoad):
                            time.sleep(0.5)
                    st.success(strFillAutomaticConfirmed)
                    with st.expander("Why I did it?"):
                        st.write(explanationWhy)
                        k = nullNum + distinct
                        if st.checkbox(strFillAutomaticRollback, key=k) == True:
                            droppedList = droppedList[ : -1]
                        else:
                            if x == 0:
                                dfAutomatic[col].fillna(dfAutomatic[col].mode(), inplace=True)
                            elif x == 1:
                                dfAutomatic[col].fillna(avgValue, inplace=True)
                st.markdown("---")

        #Checking and removing redundancies in the data
        length = round(len(dfAutomatic.index)/10)
        limit = round(length * 60 / 100)
        redundancyList = []
        for col in dfAutomatic.columns:
            for col1 in dfAutomatic.columns:
                if col != col1:
                    dup = 0
                    for i in range(length):
                        if str(dfAutomatic[col][i]) in str(dfAutomatic[col1][i]):  #col1/arg1 ubicazione, col/arg descrizioneVia
                            dup += 1
                    if dup > limit:
                        #st.write(f"The column  **{col1}** cointans the ", "%0.2f" %(percentageDup), "%" + " of the information present in the column " + f"**{col}**")
                        redundancyList.append([col, col1])
        intk = 200
        flag = 0
        for item in redundancyList:
            flag = 1
            strRemoveRedLoad = "Removing the redundancy of information between column " + item[0] + " and " + item[1]
            strRemoveRedConfirmed = f"Successfully removed all the redundancy of information between **{item[0]}** and **{item[1]}**! Now the information is present only in column **{item[0]}**."
            strRemoveRedRollback = f"Check to restore the information in column **{item[1]}**"
            if st.session_state['once'] == True:
                    with st.spinner(text=strRemoveRedLoad):
                        time.sleep(1)
            st.success(strRemoveRedConfirmed)
            with st.expander("Why I did it?"):
                st.write("The two columns were partially representing the same instances. So the redundant information was dropped from the most complete column. This because it's usually best practise to do not aggregate too much information within only one column.")
                if st.checkbox(strRemoveRedRollback, key=intk) == True:
                    droppedList = droppedList[ : -1]
                else:
                    for i in range(len(dfAutomatic.index)):
                        if str(dfAutomatic[item[0]][i]) in str(dfAutomatic[item[1]][i]):
                            try:
                                dfAutomatic[item[1]][i] = str(dfAutomatic[item[1]][i]).replace(str(dfAutomatic[item[0]][i]), "")
                                intk += 1                            
                            except:
                                intk += 1
                

        st.info("No other actions to be perfomed")
        st.markdown("---")
        st.subheader("New dataset real time preview")
        st.write(dfAutomatic)
        st.session_state['newdf'] = dfAutomatic.copy()
        st.warning("If you see columns with poor information you've the chance to drop them. Remind that you're also applying *permanently* all the changes above.")
        colSave, colSaveDrop, colIdle = st.columns([1,1,8], gap='small')
        with colSave:
            if st.button("Save", on_click=clean2):
                ()
                #st.session_state['y'] = 2
                #st.session_state['toBeProfiled'] = True
                #st.experimental_rerun()
        with colSaveDrop:
            if st.button("Save and go to Drop", on_click=clean3):
                ()
                #st.session_state['y'] = 3
                #st.experimental_rerun()
        st.session_state['once'] = False
        st.markdown("---")
        if st.button("Back To Homepage"):
            switch_page("homepage")

#elif st.session_state['y'] == 3:
if st.session_state['y'] == 3:
    dfAutomatic = st.session_state['newdf']
    listCol = dfAutomatic.columns
    listCol = listCol.insert(0, "Don't drop anything")
    colToDrop = st.multiselect("Select one or more column to drop", listCol, "Don't drop anything")
    if (len(colToDrop) > 0) and (colToDrop.count("Don't drop anything") == 0):
        #if st.button("Save"):
        #    st.session_state['newdf'] = dfAutomatic.copy()
        #    st.session_state['y'] = 2
        #    st.session_state['toBeProfiled'] = True
        #    st.experimental_rerun()
    #else:
        for col in colToDrop:
            dfAutomatic = dfAutomatic.drop(col, axis=1)
    st.subheader("Real time preview")
    st.write(dfAutomatic.head(50))
    col1, col2, col3 = st.columns([1,1,8], gap='small')
    with col1:
        if st.button("Save"):
            st.session_state['newdf'] = dfAutomatic.copy()
            st.session_state['y'] = 2
            st.session_state['toBeProfiled'] = True
            st.experimental_rerun()
    with col2:
        if st.button("Back"):
                st.session_state['y'] = 0
                st.session_state['once'] = True
                st.experimental_rerun()



elif st.session_state['y'] == 2:
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
        if st.button("Dataset Info"):
            switch_page("dataset_info")
#st.markdown("---")


