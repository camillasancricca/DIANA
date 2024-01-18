import time
import numpy as np
import pandas as pd
import streamlit as st
import random
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.no_default_selectbox import selectbox
import os
from pandas_profiling import *
import json
import numpy as np



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


def update_selectbox_style():
    st.markdown(
        """
        <style>
            .stSelectbox [data-baseweb="select"] div[aria-selected="true"] {
                white-space: normal; overflow-wrap: anywhere;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

update_selectbox_style()
update_multiselect_style()

m = st.markdown("""
<style>
div.stButton > button:first-child {
    line-height: 1.2;
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

def color_survived(val):
    color = 'palegreen'
    return f'background-color: {color}'

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


dfCol = st.session_state['arg'] #dfCol is a series here
profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
string = "Managing categories of column " + dfCol.name
st.title(string)
col1, col2, col3 = st.columns(3, gap='small')
distinctList = pd.unique(dfCol)
distinctNum = len(distinctList)
percentageDistinct = distinctNum/len(dfCol.index)*100
with col1:
    st.subheader("Preview")
    st.write(dfCol.head(50))
with col2:
    if distinctNum != 1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("This column has ", distinctNum, " different distinct values (" + "%0.2f" %(percentageDistinct) + "%)")
    else:
        st.write("This column has only one unique value that is: ", dfCol[2])
list = report["variables"][dfCol.name]["value_counts"]
with st.expander("Value counter"):
    st.write(list)

with st.expander("Values replacer"):
    if st.session_state['y'] == 0:  #choose the values to replace
        #old_values = []
        #new_value = []
        CdistinctList = distinctList.copy()
        CdistinctList = np.insert(CdistinctList, 0, "Custom Value")
        CdistinctList = np.insert(CdistinctList, 0, "--")
        new_value = st.selectbox("Final value", CdistinctList)
        if new_value == "Custom Value":
            new_value = st.text_input("Insert a custom final value")
        predefinedList = []
        if new_value != "":
            for i in range(len(distinctList)):
                if (str(new_value) != str(distinctList[i])) and ((str(new_value).lower() in str(distinctList[i]).lower()) or (str(distinctList[i]).lower() in str(new_value).lower())):
                    predefinedList.append(distinctList[i])
        old_values = st.multiselect("Replace all these values, matching categories will be provided by default",distinctList,predefinedList)
        st.session_state['old_values'] = old_values
        st.session_state['new_value'] = new_value
        valueButton = False
        if new_value in old_values:
            errorString = "You're choosing to replace " + str(new_value) + " with " + str(new_value) + ", try again"
            st.error(errorString)
            st.session_state['bool'] = True
        else:
            valueButton = st.button("Preview", key=1)
            if valueButton:
                st.session_state['y'] = 1
                st.experimental_rerun()
    elif st.session_state['y'] == 1:  #show the preview
        
        changedValues = newDistinctNum = newPercentageDistinct = 0
        old_values = st.session_state['old_values']
        new_value = st.session_state['new_value']
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            st.write("Old column")
            st.write(dfCol)
            
        with col1_2:
            indexList = []
            dfColCopy = dfCol.copy(deep=True)
            if dfCol.dtype == object:
                for i in range(len(dfCol.index)):
                    if str(dfCol[i]) in old_values:
                        dfColCopy[i] = new_value
                        changedValues += 1
                        indexList.append(i)
            elif dfCol.dtype.kind == 'f':
                for i in range(len(dfCol.index)):
                    if dfCol[i] in old_values:
                        dfColCopy[i] = new_value
                        changedValues += 1
                        indexList.append(i)
            #st.write(indexList)
            st.session_state['dfColCopy'] = dfColCopy
            st.write("New column")
            st.dataframe(dfColCopy.to_frame().style.apply(lambda x: ['background-color: lightgreen' if (indexList.count(x.name))else '' for i in x], axis=1))
            #st.write(dfColCopy.head(20))
            newDistinctList = pd.unique(dfColCopy)
            newDistinctNum = len(newDistinctList)
            newPercentageDistinct = newDistinctNum/len(dfCol.index)*100
        strChanged = str(changedValues)
        successString = strChanged + " values has been replaced successfully with " + str(new_value)
            
            #TODO valutare st.metric!!

        message1 = st.empty()
        message1.success(successString)
        st.write("New number of distinct values is ", newDistinctNum)
        st.write("New percentage of distict values is " + "%0.2f" %(newPercentageDistinct) + "%")
        radio = st.radio("Do you want to apply these changes?", ["", "No", "Yes"], horizontal=True, index=0, key = 8)
        if radio == "Yes":
            st.session_state['toBeProfiled'] = True
            st.session_state['y'] = 2
            st.experimental_rerun()
        elif radio == "No":
            st.session_state['y'] = 0
            st.experimental_rerun()

    elif st.session_state['y'] == 2:  

        #TODO
        #update the dataframe
        #do the profile again
        #remove the old report
        #load the new one
        successMessage = st.empty()
        dfColCopy = st.session_state['dfColCopy']
        df[dfColCopy.name] = dfColCopy.values
        st.subheader("Preview")
        st.dataframe(df.head(50).style.applymap(color_survived, subset=[dfColCopy.name]))
        successMessage.success("Replacement completed! Please wait while the dataframe is profiled again..")
        if st.session_state['toBeProfiled'] == True:
            profileAgain(df)
        st.session_state['toBeProfiled'] = False
        successMessage.success("Profiling updated!")
        if st.button("Replace other values"):
            st.session_state['y'] = 0
            st.experimental_rerun() 
    else:
        ()
if st.button("Back to Dataset Info!"):
        switch_page("dataset_info")

