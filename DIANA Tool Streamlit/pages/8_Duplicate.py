import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_pandas_profiling import st_profile_report
import recordlinkage
from recordlinkage.index import Block
from sklearn.metrics.pairwise import cosine_similarity
import jaro
import os
from ydata_profiling import ProfileReport
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
    newColumns = [item for item in df.columns]
    st.session_state['dfCol'] = newColumns

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

fd_data = st.session_state.setdefault("duplicate_detenction",{})
if 'setCompare' not in st.session_state:
    st.session_state['setCompare'] = []
if 'candidate_links' not in st.session_state:
    st.session_state['candidate_links'] = []

if 'threshold' not in st.session_state:
    st.session_state['threshold'] = 0.0
if 'weights' not in st.session_state:
    st.session_state['weights'] = []
if 'dfPreview' not in st.session_state:
    st.session_state['dfPreview'] = None
if 'Once' not in st.session_state:
    st.session_state['Once'] = False
if 'droppedRow' not in st.session_state:
    st.session_state['droppedRow'] = []

st.title("Duplicates detection")
slate1 = st.empty()
body1 = slate1.container()
slate2 = st.empty()
body2 = slate2.container()

df = st.session_state['df']
if 'my_dataframe' not in st.session_state:
    st.session_state.my_dataframe = df

st.write(st.session_state.my_dataframe)

with body1:
    df = st.session_state['df']
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df

    setColToDropWeights = set()
    with st.expander("Expand to better understand the techniques used to detect duplicates and calculate the similarity between two rows.", expanded=False):
        st.write(f"The technique used for **record linkage** is the blocking technique. The dataset will be divided into blocks, where each block consists of tuples that have the same values for the selected attribute or attributes below..")
        st.write(f"For **similarity** the Jaro-Winkler metric is used. It is a string edit distance between two sequences. In this case, the algorithm is slightly modified: to calculate the similarity between two rows, the blocking attributes and attributes explicitly removed with the appropriate checkbox will not be used. For the remaining attributes, in case their checkbox is selected, they will have a weight of 2, while those not selected will have a weight of 1. Therefore, to calculate similarity, a weighted average will be computed..")


    listCol = st.session_state.my_dataframe.columns if st.session_state.my_dataframe is not None else []

    colToDrop = st.multiselect("Select one or more columns that will be used to search for duplicates.", listCol)


    # Altri passaggi del codice...

    # threshold selection, sarebbe bello con uno slider -> attenzione al refresh(potrebbe essere meglio usare quello di stremlit)
    numOfCouples = 0
    index = st.session_state.my_dataframe.index
    max = ((len(index) * (len(index) - 1)) / 5)
    if len(colToDrop) > 0:
        try:
            Blocker = Block(on=colToDrop)
            candidate_links = Blocker.index(st.session_state.my_dataframe)
            numOfCouples = len(candidate_links)
        except:
            st.error("Select something.")
        if numOfCouples >= max:
            st.error(f"This set is not suitable for blocking because it generates too many comparisons. (**{numOfCouples}**). Modify the blocking set.")
        elif 0 < numOfCouples < max:
            st.info(f"exist **{numOfCouples}** pairs of rows that have the same values for this set of columns. Here are two examples:")
            st.write(st.session_state.my_dataframe.iloc[[candidate_links[1][0], candidate_links[1][1]]].style.apply(
                lambda x: ['background-color: lightgreen' if x.name in colToDrop else '' for i in x], axis=0))
            st.write(st.session_state.my_dataframe.iloc[[candidate_links[3][0], candidate_links[3][1]]].style.apply(
                lambda x: ['background-color: lightgreen' if x.name in colToDrop else '' for i in x], axis=0))
            setCompareTemp = list(st.session_state.my_dataframe.columns.drop(colToDrop))
            weights = []
            setCompare = []
            removedCols = []
            for col in st.session_state.my_dataframe.columns:
                if col in setCompareTemp and (
                        len(pd.unique(st.session_state.my_dataframe[col])) == len(st.session_state.my_dataframe.index) or len(
                        pd.unique(st.session_state.my_dataframe[col])) == 1):
                    setCompareTemp.remove(col)
                    removedCols.append(col)
                else:
                    ()
            st.markdown("---")
            st.write(
                f"Automatic removal from the set used for similarity calculation of all columns that have unique values or different values for each row, in this case **{', '.join(column for column in removedCols)}**.\n\nThis is because the similarity parameter for these attributes will always be 1 or 0 in any case and will not add further information. Additionally, the attributes used for blocking have been automatically removed: since they are the same within a pair, it makes no sense to calculate the distance as it will surely be 1.")
            colToDropWeights = st.multiselect(
                "Select one or more additional columns that will not be used for calculating the similarity between two rows.",
                setCompareTemp)
            if len(colToDropWeights) > 0:
                setColToDropWeights = set(colToDropWeights)
                setCompare = [x for x in setCompareTemp if x not in setColToDropWeights]
            else:
                setCompare = setCompareTemp
            st.markdown("---")
            st.subheader("Select the most important attributes")
            st.info(
                "The selected attributes will have twice the weight compared to the non-selected ones. In case nothing is selected, each attribute will have the same weight.")
            st.write("")
            columnsWeight = st.columns([1 for i in range(len(setCompare))], gap="small")
            for i in range(len(setCompare)):
                with columnsWeight[i]:
                    label = str(setCompare[i])
                    if st.checkbox(f"**{label}**", key=i):
                        weights.append([2, label])
                    else:
                        weights.append([1, label])
            st.markdown("---")
            threshold = st.slider("Select a similarity threshold above to display pairs of duplicate rows",
                                  0.01, 1.00, value=0.9, step=0.01)
            if threshold < 0.9:
                st.warning("Be cautious as the calculation may take a few minutes!")
            st.write("")
            drop = st.checkbox("Select to remove as many rows as possible without losing any information. This action will not permanently modify the dataset; you will need to save it explicitly on the next page.")
            st.write("")
            st.write("")
            st.session_state['weights'] = weights
            st.session_state['threshold'] = threshold
            st.session_state['setCompare'] = setCompare
            st.session_state['candidate_links'] = candidate_links
            st.session_state['drop'] = drop

            columns = st.columns([1, 10, 1], gap='small')
            with columns[0]:
                if st.button("GO!"):
                    ()
            with columns[2]:
                if st.button("Delete selection"):
                    ()
        elif len(colToDrop) > 0:
            st.error("One of the selected attributes is not suitable for blocking, as it is unique for each row..")
    else:
        ()

with body2:
    candidate_links = st.session_state['candidate_links']
    setCompare = st.session_state['setCompare']
    threshold = st.session_state['threshold']
    weights = st.session_state['weights']
    dfPreview = st.session_state.my_dataframe.copy()
    i = 0
    count = 0
    changed = 0
    st.session_state['droppedRow'] = []
    with st.form("form"):
        for item in candidate_links:
            i += 1
            row2Null = st.session_state.my_dataframe.iloc[item[0]].isna().sum(axis=0)
            row1Null = st.session_state.my_dataframe.iloc[item[1]].isna().sum(axis=0)
            jaroNum = 0
            totalWeight = 0
            for col, weight in zip(setCompare, weights):
                if str(weight[1]) == col:
                    totalWeight += weight[0]
                    jaroNum += (
                            jaro.jaro_winkler_metric(str(st.session_state.my_dataframe.iloc[item[0]][col]),
                                                     str(st.session_state.my_dataframe.iloc[item[1]][col])) * weight[0])
            sim = jaroNum / totalWeight
            if sim == 1:
                st.write("The similarity of couple ", i, " is ", sim)
                st.write(st.session_state.my_dataframe.iloc[[item[1], item[0]]])
                try:
                    count += 1
                    changed += 1
                    st.write("Since these two rows are identical, the second one is arbitrarily eliminated. No information is lost.")
                    dfPreview.drop([item[0]], axis=0, inplace=True)
                except:
                    ()
                st.markdown("---")
            elif sim >= threshold:
                st.write("The similarity of couple ", i, " is ", sim)
                st.write(st.session_state.my_dataframe.iloc[[item[1], item[0]]])
                count += 1
                if (row1Null + row2Null) >= 0:
                    if drop:
                        result = {}
                        row1 = st.session_state.my_dataframe.iloc[item[0]]
                        row2 = st.session_state.my_dataframe.iloc[item[1]]
                        flag = "YES"
                        for attr in setCompare:
                            if str(row1[attr]) in ['<NA>', 'nan']:
                                result[attr] = row2[attr]
                            elif str(row2[attr]) in ['<NA>', 'nan']:
                                result[attr] = row1[attr]
                            elif str(row1[attr]) == str(row2[attr]):
                                result[attr] = row1[attr]
                            else:
                                flag = "NO"
                                break
                        if flag == "YES":
                            try:
                                if row1Null > row2Null:
                                    dfPreview.loc[item[0], setCompare] = [result[col] for col in setCompare]
                                    st.write(dfPreview.loc[[item[0]]])
                                    dfPreview.drop([item[1]], axis=0, inplace=True)
                                    changed += 1
                                else:
                                    dfPreview.loc[item[1], setCompare] = [result[col] for col in setCompare]
                                    st.write(dfPreview.loc[[item[1]]])
                                    dfPreview.drop([item[0]], axis=0, inplace=True)
                                    changed += 1
                                st.write("The two rows above will be replaced with this one. No information is los")
                            except:
                                ()
                        else:
                            choice = st.radio(f"Select (Riga {i})", ("Nobody", "Delete first row", "Delete second row"),
                                              key=f"radio_{i}")
                            st.write("")
                            if choice == "Delete first row":
                                st.session_state['droppedRow'].append(item[1])
                            elif choice == "Delete second row":
                                st.session_state['droppedRow'].append(item[0])
                            else:
                                st.session_state['droppedRow'].append(-1)
                        st.markdown("---")
                st.markdown("---")
        st.info(
            f"Above the similarity threshold of {threshold}, there are {count} pairs of similar rows to display. Of these, **{changed}** righe sono state eliminate.")
        st.warning("This action will be permanent, the dataset will also be profiled again. Wait for the new dataset to appear.")
        st.write("")
        save_button = st.form_submit_button(label="Save")
        if save_button:
            for item in st.session_state['droppedRow']:
                if item != -1:
                    dfPreview = dfPreview.drop(item)
            dfPreview = dfPreview.reset_index(drop=True)
            st.session_state.my_dataframe = dfPreview
            st.subheader("New dataset")
            st.write(st.session_state.my_dataframe)

st.write("---")

if st.button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Download")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Cleaning")



