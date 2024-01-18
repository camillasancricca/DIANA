import json
import time
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

df = st.session_state['df']
dfCol = st.session_state['dfCol']
profile = st.session_state['profile']
report = st.session_state['report']

#df["Civico"] = pd.to_numeric(df["Civico"], errors='coerce')
#df['Ubicazione']=df['Ubicazione'].astype('category').cat.codes
#df['DescrizioneVia']=df['DescrizioneVia'].astype('category').cat.codes
#df['MUNICIPIO']=df['MUNICIPIO'].astype(str)
#df['NIL']=df['NIL'].astype('category').cat.codes
#df['Location']=df['Location'].astype('category').cat.codes

#if st.session_state['status'] == 0:
st.title("Dataset information")
st.subheader("Preview")
st.write(df.head(50))
with st.expander("Incomplete Rows", expanded=False):
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
        st.subheader("Preview")
        st.dataframe(dfToDelete)
        if st.button("Drop these rows from the dataset", key=-1):
            st.session_state['toBeProfiled'] = True
            df.drop(nullToDelete, axis=0, inplace=True)
            successMessage = st.empty()
            successString1 = "Dataset successfully updated! Please wait while the dataframe is profiled again.."
            successMessage.success(successString1)
            if st.session_state['toBeProfiled'] == True:
                profileAgain(df)
            st.session_state['toBeProfiled'] = False
            successMessage.success("Profiling updated!")
            time.sleep(2)
            st.experimental_rerun()
    else:
        numm = colNum - threshold 
        successString = "The dataset has all the rows with at least " + str(numm) + " not null values out of " + str(len(df.columns))
        st.success(successString)
colMain1, colMain2, colMain3, colMain4 = st.columns([1, 0.001, 0.05, 0.4])
with colMain1:
    st.title("Single column")
    with st.expander("Single Column Analysis' Legend"):
        st.write("Column type: is the format of the column")
        st.write("Null values is the number of null values within the column")
        st.write("Distinct values is the number that indicates how many distinct occurences are present in the column. For example [1, 2, 2, 3] has 3 distinct values: 1,2,3")
        st.write("Unique values is the number that indicates how many unique occurences are present in the column. For example [1, 2, 2, 3] has only 2 unique value that are 1 and 3. Intuitively, if a column doesn't have a lot of unique values it means that the column won't be a useful candidate for the primary key role.")
        st.write("")

    with st.expander("Single Column Analysis"):
        count = 0
        distinctButton = nullButton = 0
        for col in dfCol:
            nullNum = df[col].isna().sum()
            distinctNum = len(pd.unique(df[col]))
            testCount = df[col].drop_duplicates().size
            if report["variables"][dfCol[int(count)]]["type"] != "Variable.S_TYPE_UNSUPPORTED":

                columnUnique = report["variables"][dfCol[int(count)]]["n_unique"]
                percentageUnique = columnUnique/len(df.index)*100
            else:
                columnUnique = "Not available"
            st.title(col)
            st.write("Column type is: ",str(df[col].dtype))
            st.write("Null values: ",nullNum)
            percentageNull = nullNum/len(df.index)*100
            st.write("Percentage of null values: ","%0.2f" %(percentageNull) + "%")
            if percentageNull > 25:
                nullButton += randrange(1000)
                col1, col2, col3 = st.columns(3,gap="small")
                with col1:
                    st.error("This attribute has more than 25" + "%" + " of null values")
                with col2:
                    key1 = count + nullButton
                    if st.button("Handle null values", key=key1):
                        st.session_state['from'] = 0
                        st.session_state['y'] = 0
                        st.session_state['arg'] = df[col].copy(deep=False)
                        switch_page("null_values")
            st.write("Distinct values: ", distinctNum)
            percentageDistinct = distinctNum/len(df.index)*100
            st.write("Percentage of distinct values: ","%0.2f" %(percentageDistinct) + "%")
            if percentageDistinct < 4:
                #distinctButton += randrange(1000)
                col1, col2, col3 = st.columns(3,gap="small")
                with col1:
                    st.warning("This attribute is almost a category for the dataset")
                    #st.write(type(dfnew[df[col]]))
                with col2:
                    distinctButton = count + 1
                    if st.button("Manage categories", key=distinctButton):
                        st.session_state['y'] = 0
                        st.session_state['arg'] = df[col].copy(deep=False)
                        switch_page("category")
            st.write("Unique values: ", columnUnique)
            if columnUnique != "Not available":
                st.write("Percentage of unique values: ","%0.2f" %(percentageUnique) + "%")
                if percentageUnique > 95:
                    st.warning("This attribute is a possible candidate for primary key!")
            count += 1

    st.title("Data dependencies and possible redundancies")
    with st.expander("Possible Redundancies in column names"):
        name_counter = 100
        for col in dfCol:
            for col1 in dfCol:
                if col != col1 and col in col1:
                    st.write(f"**{col}** and **{col1}** can potentially represent the same data given their similar column name")
                    name_counter += 1
                    ##TODO
                    #columns here e metti come opzione anche il column renaming (hai gia due argomenti, da capire il session state di col renaming)
                    if st.button("Manage", key=name_counter):
                        st.session_state['y'] = 0
                        st.session_state['arg'] = df[col].copy(deep=False)
                        st.session_state['arg1'] = df[col1].copy(deep=False)
                        switch_page("name_redundancy")
                    name_counter += 1
                    if st.checkbox("Show sample", key=name_counter):
                        st.write("Here are shown the first 30 lines of the two columns")
                        st.write(df[[col, col1]].head(30))

    with st.expander("Possible redundancies in the data"):
        st.warning("As sample for this test has been used the first 10 " + "%" + "of the dataset.")
        length = round(len(df.index)/10)
        limit = round(length * 90 / 100)
        data_counter = df.size
        for col in dfCol:
            for col1 in dfCol:
                if col != col1:
                    dup = 0
                    for i in range(length):
                        if str(df[col][i]) in str(df[col1][i]):
                            #st.write(df[col][i])
                            dup += 1
                    if dup > limit:
                        percentageDup = dup / length * 100
                        #st.write(limit)
                        st.write(f"The column  **{col1}** cointans the ", "%0.2f" %(percentageDup), "%" + " of the information present in the column " + f"**{col}**")
                        data_counter += 1
                        if st.button("Manage", key=data_counter):
                            st.session_state['y'] = 0
                            st.session_state['arg'] = df[col].copy(deep=False)
                            st.session_state['arg1'] = df[col1].copy(deep=False)
                            switch_page("data_redundancy")
                        data_counter += 1
                        if st.checkbox("Show sample", key=data_counter):
                            st.write("Here are shown the first 30 lines of the two columns")
                            st.write(df[[col1, col]].head(30))
                            

    correlations = profile.description_set["correlations"]
    phik_df = correlations["phi_k"]
    #st.title("df.corr")
    #st.write(df.corr())
    st.title("Correlation's table")
    with st.expander("More info"):
        st.write("""
        This is the **Phi_k** correlation table. Why I've choosen this table? Phi_K is a new and practical correlation coefficient based on several refinements to Pearson’s hypothesis test of independence of two variables.
        The combined features of Phi_K form an advantage over existing coefficients. First, it works consistently between categorical, ordinal and interval variables. Second, it captures non-linear dependency. Third, it reverts to the Pearson correlation coefficient in case of a bi-variate normal input distribution. Anyway a weak point of this table is that is very expensive from a computational point of view.                                                                                                                                                          
        Docs available [here](https://phik.readthedocs.io/en/latest/)
    """)
    st.write(phik_df.style.background_gradient())
    #st.write(df.head(100))
    with st.expander("What does the correlation between two columns means?"):
        st.write("The correlation between two columns is defined as how much values of a column are related with values of another column. In other words we can explain it as: given the value of the column X, with how much confidence I can guess the value on column Y, given also the previous observations of X->Y?")
    ind = 1
    st.subheader("List of the correlations that are higher than 85%")
    corrButton = name_counter + 1
    correlationList = []
    for col in phik_df.columns:
        if ind < (len(phik_df.columns) - 1):
            for y in range(ind, len(phik_df.columns)):
                x = float(phik_df[col][y])*100
                if x > 85:
                    col1, col2, col3 = st.columns([2, 1.5, 6], gap="small")
                    with col1:
                        #st.write(f"Column name is ")
                        st.write(f"The correlation between  **{col}** and **{str(phik_df.columns[y])}**  is: ", "%0.2f" %(x) , "%")
                        if x > 85:
                            correlationList.append([col, str(phik_df.columns[y]), x])
                    with col2:
                        if st.button("Manage", key=corrButton):
                            st.session_state['y'] = 0
                            st.session_state['from'] = 0
                            st.session_state['correlation'] = x
                            st.session_state['arg'] = df[col].copy(deep=False)
                            st.session_state['arg1'] = df[phik_df.columns[y]].copy(deep=False)
                            switch_page("data_correlation")
                    corrButton += 1
                    st.markdown("---")
                    #string1 = '**' + col + '**'
                    #st.markdown(string1)
                    #string = "The correlation between " + col + " and " + str(phik_df.columns[y]) + " is: " + "%0.2f" %(x) + "%"
                    #st.warning(string)
            ind += 1
    correlationSum = {}
    for y in range(0, len(phik_df.columns)):
        x = 0
        z = 0
        for column in phik_df.columns:
            #st.write(tttt)
            #st.write(y)
            z = float(phik_df[column][y])
            x += z 
        #st.write(f"{str(phik_df.columns[y])} , {x}")
        correlationSum.update({str(phik_df.columns[y]) : x}) 
with colMain2:
    ()
with colMain3:
    ()
with colMain4:
    ()
        #if st.button("Hi"):
        #    webbrowser.open("http://localhost:8501/dataset_info#automatic", 0)
#elif st.session_state['status'] == 1:
#    st.write("Test")
if st.button("Homepage"):
        switch_page("Homepage")