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
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import missingno as msno
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_extras.no_default_selectbox import selectbox

st.set_page_config(page_title="Transformation", layout="wide", initial_sidebar_state="collapsed")

df = st.session_state['df']

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
    border: 1px solid white;
    border-radius: 5px;
    padding: 10px;
    font-size: 2rem;
    color: grey; 
    font-family: 'Verdana'
}
</style>""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;  /* Cambia il colore dello sfondo qui */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<style>h1{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)
st.markdown("<style>h3{color: black; font-family: 'Verdana', sans-serif;}</style>", unsafe_allow_html=True)


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

profile = st.session_state['profile']
report = st.session_state['report']
df = st.session_state['df']
dfCol = st.session_state['dfCol']

st.session_state['y'] = 0



st.title("Data Transformation")

pages = {
    "Dataset": "pagina_1",
    "Values Editing": "pagina_2",
    "Column Renaming": "pagina_3",
    "Column Splitting" : "pagina_4",
    "Columns Merging" : "pagina_5",
    "Column Dropping" : "pagina_6"
}

# Crea un menu a tendina nella barra laterale
scelta_pagina = st.sidebar.selectbox("Select a page:", list(pages.keys()))


if scelta_pagina == "Dataset":
    st.subheader("Dataset")
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.write(st.session_state.my_dataframe)
    st.write("---")




elif scelta_pagina == "Values Editing":
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
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

    if 'profile' not in st.session_state:
        st.session_state['profile'] = None
    if 'report' not in st.session_state:
        st.session_state['report'] = None
    df = st.session_state.my_dataframe
    dfCol = st.session_state.my_dataframe.columns

    st.title("Edit column values")
    with st.expander("Manual", expanded=False):
        st.write("In this page you're allowed to select a column in which apply some splitting/changes to its values")
        column = selectbox("Choose a column", dfCol)
        if column != None:
            unique = st.session_state.my_dataframe.duplicated(subset=column).value_counts()[0]
            st.write("Unique rows: ", unique)
            st.write("Duplicate rows", len(st.session_state.my_dataframe.index) - unique)
            # st.write("Unique rows: ", df.duplicated(subset=column).value_counts())
            col1, col2, col3 = st.columns(3, gap="small")
            with col1:
                st.write('Old column')
                st.write(st.session_state.my_dataframe[column])
            with col2:
                if st.session_state['y'] == 0:
                    st.write("Edit menu")
                    action = selectbox("Action", ["Remove", "Add"])
                    if action == "Add":
                        where = selectbox("Where", ["Start of the value", "End of the value"])
                        if where == "Start of the value":
                            inputString = st.text_input(
                                "Provide the string (remember, if needed, the space at the end)")
                            st.session_state['string'] = inputString
                            if st.button("Go!"):
                                st.session_state['y'] = 1
                                st.experimental_rerun()
                        elif where == "End of the value":
                            inputString = st.text_input(
                                "Provide the string (remember, if needed, the space before type the string)")
                            st.session_state['string'] = inputString
                            if st.button("Go!"):
                                st.session_state['y'] = 2
                                st.experimental_rerun()
                    elif action == "Remove":
                        inputString = st.text_input("Provide the string")
                        st.session_state['string'] = inputString
                        if st.button("Go!"):
                            st.session_state['y'] = 3
                            st.experimental_rerun()
                    else:
                        ()
                elif st.session_state['y'] == 1:  # add start
                    string = st.session_state['string']
                    copyPreview = st.session_state.my_dataframe[column].copy()
                    for i in range(len(df.index)):
                        copyPreview[i] = string + str(copyPreview[i])
                    st.session_state['copyPreview'] = copyPreview
                    st.session_state['y'] = 4
                    st.experimental_rerun()
                elif st.session_state['y'] == 2:  # add end
                    string = st.session_state['string']
                    copyPreview = st.session_state.my_dataframe[column].copy()
                    for i in range(len(st.session_state.my_dataframe.index)):
                        copyPreview[i] = str(copyPreview[i]) + string
                    st.session_state['copyPreview'] = copyPreview
                    st.session_state['y'] = 4
                    st.experimental_rerun()
                elif st.session_state['y'] == 3:  # remove
                    string = st.session_state['string']
                    copyPreview = st.session_state.my_dataframe[column].copy()
                    for i in range(len(st.session_state.my_dataframe.index)):
                        if string in str(copyPreview[i]):
                            copyPreview[i] = str(copyPreview[i]).replace(string, '')
                    st.session_state['copyPreview'] = copyPreview
                    st.session_state['y'] = 4
                    st.experimental_rerun()
                elif st.session_state['y'] == 4:
                    st.write("New column")
                    copyPreview = st.session_state['copyPreview']
                    st.write(copyPreview)
                    columns = st.columns([1, 1, 4])
                    with columns[0]:
                        if st.button("Save"):
                            st.session_state['toBeProfiled'] = True
                            st.session_state['y'] = 5
                            st.experimental_rerun()
                    with columns[1]:
                        if st.button("Back"):
                            st.session_state['y'] = 0
                            st.experimental_rerun()

        else:  # column is none
            ()
        if st.session_state['y'] == 5:
            copyPreview = st.session_state['copyPreview']
            st.session_state.my_dataframe[copyPreview.name] = copyPreview.values
            successMessage = st.empty()
            successString = "Column successfully updated! Please wait while the dataframe is profiled again.."
            successMessage.success(successString)
            if st.session_state['toBeProfiled'] == True:
                profileAgain(st.session_state.my_dataframe)
            st.session_state['toBeProfiled'] = False
            st.session_state['y'] = 1
            successMessage.success("Profiling updated!")
            newUnique = copyPreview.duplicated().value_counts()[0]
            st.write("Unique rows of the new column: ", newUnique)
            st.write("Duplicate rows of the new column: ", len(st.session_state.my_dataframe.index) - newUnique)
            # st.write("Non unique rows: ", copyPreview.duplicated().sum())
            if st.button("Continue with the filtering"):
                st.session_state['y'] = 0
                st.experimental_rerun()
                # st.write("Non unique rows: ", df.duplicated(subset=copyPreview.name).value_counts()[1])
                # st.write("Unique rows: ", df.duplicated(subset=copyPreview.name).value_counts()[0])

    with st.expander("Automatic", expanded=True):
        st.write("")
        if st.session_state['y'] == 0:
            st.write(
                "In this page you're allowed to select a column in which apply some splitting/changes to its values")
            column = selectbox("Select a column", dfCol)
            delimiters = [",", ";", " ' ", "{", "}", "[", "]", "(", ")", " \ ", "/", "-", "_", ".", "|"]
            if column != None:
                counter = 0
                importantDelimiters = []
                copyPreview = st.session_state.my_dataframe[column].copy()
                unique = st.session_state.my_dataframe.duplicated(subset=column).value_counts()[0]
                st.write("")
                col1_A, col2_A, col3_A, col4_A = st.columns([1, 1, 1, 6.2])
                actions = []
                # st.write("Edit menu")
                for item in delimiters:
                    counter = 0
                    for i in range(int(len(st.session_state.my_dataframe.index) / 10)):  # search for a delimiter in the first 10% of the dataset
                        x = str(copyPreview[i]).find(item)
                        if x > 0:
                            counter += 1
                    if counter >= (
                            len(st.session_state.my_dataframe.index) / 50):  # if in the first 10%, is present at least the 20% then we should propose it!
                        importantDelimiters.append(item)
                # st.write(importantDelimiters)
                for item in importantDelimiters:
                    stringToAppend = "Cut everything before " + item
                    actions.append(stringToAppend)
                    stringToAppend = "Cut everthing after " + item
                    actions.append(stringToAppend)
                with col1_A:
                    st.write("")
                    st.write("Unique rows: ", unique)
                with col2_A:
                    st.write("")
                    st.write("Duplicate rows: ", len(df.index) - unique)
                with col4_A:
                    if len(actions) == 0:
                        st.error("It's not possible to split this column, as no delimiters are detected")
                        action = []
                    else:
                        action = selectbox("Choose an action", actions)
                # st.write("Unique rows: ", df.duplicated(subset=column).value_counts())
                col1, col2, col3 = st.columns(3, gap="small")
                with col1:
                    st.write('Old column')
                    st.write(st.session_state.my_dataframe[column])
                with col2:
                    if action != None and len(actions) > 0:
                        for item in importantDelimiters:
                            if item in action:
                                if "before" in action:
                                    st.session_state['item'] = item
                                    st.session_state['action'] = "before"
                                    for i in range(len(st.session_state.my_dataframe.index)):
                                        if item in str(copyPreview[i]):
                                            tempStringList = str(copyPreview[i]).split(item, 1)
                                            if len(tempStringList) > 0:
                                                if len(tempStringList[1]) == 0:
                                                    copyPreview[i] = None
                                                else:
                                                    copyPreview[i] = tempStringList[1]
                                elif "after" in action:
                                    st.session_state['item'] = item
                                    st.session_state['action'] = "after"
                                    for i in range(len(st.session_state.my_dataframe.index)):
                                        if item in str(copyPreview[i]):
                                            tempStringList = str(copyPreview[i]).split(item, 1)
                                            if len(tempStringList) > 0:
                                                if len(tempStringList[0]) == 0:
                                                    copyPreview[i] = None
                                                else:
                                                    copyPreview[i] = tempStringList[0]

                        st.write("New column")
                        st.write(copyPreview)
                        st.session_state['copyPreview'] = copyPreview
                        newUnique = copyPreview.duplicated().value_counts()[0]
                        st.write("Unique rows of the new column: ", newUnique)
                        st.write("Duplicate rows of the new column: ", len(st.session_state.my_dataframe.index) - newUnique)
                        if st.button("Save"):
                            st.session_state['toBeProfiled'] = True
                            st.session_state['y'] = 6
                            st.experimental_rerun()
        if st.session_state['y'] == 6:
            copyPreview = st.session_state['copyPreview']
            df[copyPreview.name] = copyPreview.values
            successMessage = st.empty()
            successString = "Column successfully updated! Please wait while the dataframe is profiled again.."
            successMessage.success(successString)
            if st.session_state['toBeProfiled'] == True:
                profileAgain(st.session_state.my_dataframe)
            st.session_state['toBeProfiled'] = False
            st.session_state['y'] = 1
            successMessage.success("Profiling updated!")
            newUnique = copyPreview.duplicated().value_counts()[0]
            st.write("Unique rows of the new column: ", newUnique)
            st.write("Duplicate rows of the new column: ", len(st.session_state.my_dataframe.index) - newUnique)


elif scelta_pagina == "Column Renaming":
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df

    st.title('Rename a Column')

# Select a column to rename
    column_to_rename = st.selectbox('Select a column to rename:', st.session_state.my_dataframe.columns)

# Enter a new name for the column
    new_column_name = st.text_input('Enter the new name for the column:', value=column_to_rename)

# Button to rename the column
    if st.button('Rename Column'):
        if column_to_rename in st.session_state.my_dataframe.columns:
            st.session_state.my_dataframe = st.session_state.my_dataframe.rename(columns={column_to_rename: new_column_name})
            st.success(f'Column "{column_to_rename}" has been renamed to "{new_column_name}".')

# Display the updated DataFrame
    st.write('Updated DataFrame:')
    st.write(st.session_state.my_dataframe)
    st.write("---")
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df

    st.write("---")

elif scelta_pagina == "Column Splitting":
    st.title('Column Splitting')

# Select a column to split
    column_to_split = st.selectbox('Select a column to split:', st.session_state.my_dataframe.columns)

# Check if the selected column exists in the DataFrame
    if column_to_split in st.session_state.my_dataframe.columns:
    # Enter a delimiter for splitting
        delimiter = st.text_input('Enter a delimiter (e.g., ";"):')

    # Button to split the column
        if st.button('Split Column'):
            if delimiter:
            # Check if the selected column is of string type
                if st.session_state.my_dataframe[column_to_split].dtype == 'object':
                # Create a textbox for the name of the first new column
                    new_column1_name = st.text_input('Enter the name of the first new column', key="new_column1_name")
                    if new_column1_name:
                    # Create a textbox for the name of the second new column
                        new_column2_name = st.text_input('Enter the name of the second new column', key="new_column2_name")
                        if new_column2_name:
                            new_column1_values = st.session_state.my_dataframe[column_to_split].str.split(delimiter).str[0]
                            new_column2_values = st.session_state.my_dataframe[column_to_split].str.split(delimiter).str[1]

                        # Add the new columns to the DataFrame
                            st.session_state.my_dataframe[new_column1_name] = new_column1_values
                            st.session_state.my_dataframe[new_column2_name] = new_column2_values

                        # Remove the original column
                            st.session_state.my_dataframe.drop(column_to_split, axis=1, inplace=True)

                        # Show statistics on delimiter frequency in the column
                            delimiter_count = st.session_state.my_dataframe[new_column1_name].str.count(delimiter).sum()
                            st.write(f"Delimiter '{delimiter}' Frequency: {delimiter_count} occurrences")

                        # Show the updated DataFrame
                            st.write('Updated DataFrame:')
                            st.write(st.session_state.my_dataframe)

                        else:
                            st.warning('Enter a name for the second new column.')
                    else:
                        st.warning('Enter a name for the first new column.')

                else:
                    st.warning('The selected column is not of string type. Select a different column.')
            else:
                st.warning('Enter a valid delimiter to proceed.')
        else:
            st.warning('Select a column to split.')
    else:
        st.warning(f"Column '{column_to_split}' does not exist in the DataFrame. Select a different column.")

# Show the original DataFrame
    st.subheader('Original DataFrame:')
    st.write(st.session_state.my_dataframe)

# Button to save the updated DataFrame to session state
    if st.button('Save Updated DataFrame'):
        st.session_state.my_dataframe = st.session_state.my_dataframe
        st.success('DataFrame saved successfully!')

    st.write("---")

elif scelta_pagina == "Columns Merging":


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


    def clean1():
        slate1.empty()
        st.session_state['y'] = 1
        st.session_state['toBeProfiled'] = True


    def clean2():
        slate2.empty()
        st.session_state['y'] = 2


    profile = st.session_state['profile']
    report = st.session_state['report']
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df

    st.title("Columns merging")
    slate1 = st.empty()
    body1 = slate1.container()

    slate2 = st.empty()
    body2 = slate2.container()
    name = ""
    finalName = ""
    with body1:
        if st.session_state['y'] == 0:  # choose the values to replace

            st.subheader("Dataset preview")
            st.write(st.session_state.my_dataframe.head(200))
            columns = list(st.session_state.my_dataframe.columns)
            columns.insert(0, "Select a column")
            columnsCopy = columns.copy()
            column1 = st.selectbox("Select the first column to be merged", columns, key=1)
            if column1 != "Select a column":
                columnsCopy.remove(column1)
                column2 = st.selectbox("Select the second column to be merged", columnsCopy, key=2)
                if column2 != "Select a column":
                    name = st.selectbox(
                        "Select the name of the new column: a custom one or the name of one of the previous columns",
                        [" ", "Custom name", column1, column2])
                    if name == "Custom name":
                        finalName = st.text_input("Type the new name and press enter")
                    elif name == column1 or name == column2:
                        finalName = name
                    if finalName != "":
                        st.subheader("Preview of the new dataset")
                        columns = list(st.session_state.my_dataframe.columns)
                        colIndex = columns.index(column1)
                        if colIndex > 0:
                            colIndex -= 1
                        dfPreview = st.session_state.my_dataframe.copy()
                        dfColPreview = pd.Series()
                        # IF NOT NULL
                        dfColPreview = dfPreview[column1].astype(str) + " " + dfPreview[column2].astype(str)
                        dfPreview.drop(column1, inplace=True, axis=1)
                        dfPreview.drop(column2, inplace=True, axis=1)
                        dfPreview.insert(loc=colIndex, column=finalName, value=dfColPreview)
                        st.write(dfPreview.head(50))
                        st.warning("This action will be permanent")
                        st.session_state.my_dataframe = dfPreview.copy()
                        if st.button("Save", on_click=clean1):
                            ()
            st.markdown("---")
            if st.button("Back to Homepage"):
                switch_page("Homepage")

    if st.session_state['y'] == 1:
        if st.session_state['toBeProfiled'] == True:
            st.session_state.my_dataframe = st.session_state.my_dataframe['dfPreview']
            successMessage = st.empty()
            successString = "Please wait while the dataframe is profiled again with all the applied changes.."
            successMessage.success(successString)
            with st.spinner(" "):
                profileAgain(st.session_state.my_dataframe)
            successMessage.success("Profiling updated!")
            st.session_state['toBeProfiled'] = False
            st.subheader("New dataset")
            st.write(st.session_state.my_dataframe.head(50))
        st.markdown("---")
        if st.button("Homepage"):
            switch_page("Homepage")

elif scelta_pagina == "Column Dropping":
    st.title("Column Dropping")
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df

    # Seleziona le colonne da eliminare
    selected_columns = st.multiselect("Select the column to remove", st.session_state.my_dataframe.columns)

    if st.button("Remove Column"):
        if selected_columns:
            st.session_state.my_dataframe = st.session_state.my_dataframe.drop(selected_columns, axis=1)
            st.success(f"{len(selected_columns)} columns eliminated")

    # Mostra l'anteprima del dataset risultante
    st.subheader("Updated Dataset:")
    st.write(st.session_state.my_dataframe)


if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Outliers Inspection")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Profiling")
