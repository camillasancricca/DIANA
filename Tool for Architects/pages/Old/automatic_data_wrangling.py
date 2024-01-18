import streamlit as st
import pandas as pd
import numpy as np
from pages.data_uploading import uploading_csv
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re
import time
import statistics


def showDownloadButton(datasetToShow):
    st.sidebar.download_button(label="Download your new dataset",
                               data=datasetToShow, file_name="NewDF.csv", mime="text/csv")


def app():
    st.title("Data Wrangling App")

    #####################
    ### csv uploading ###
    #####################

    df = uploading_csv()

    if df is not None:
        profileGenerated = False

        def writeColumns():
            dfCol = []
            for col in wrangler_df.columns:
                dfCol.append(col)

        firstRows = st.checkbox("Select to show a preview of your dataset:", value=False, key=10)

        if firstRows == True:
            st.write(df.head())
        right_df = st.radio("Is this the dataset you want to work on?", ("No", "Yes"), 0, key=100)
        if right_df == "No":
            st.error("Please reload your dataset on the uploader or agree!")
        else:

            #############################
            ### creation of profiling ###
            #############################

            profilingReportBox = st.checkbox(
                "Select to generate the profiling report and start analyzing your dataset:", value=False, key=11)
            if profilingReportBox == True:
                try:

                    ##########################################################################################
                    ### ATTENTION: change the folder path where you want to create the json profile report ###
                    ##########################################################################################

                    with open("newProfile.json", 'r') as f:
                        report = json.load(f)
                except FileNotFoundError:
                    with st.spinner('Generating profiling...'):
                        time.sleep(5)
                        profile = ProfileReport(df, title="Pandas Profiling Report", progress_bar=True, minimal=True)
                        profile.to_file("newProfile.json")
                        with open("newProfile.json", 'r') as f:
                            report = json.load(f)
                    st.success('Done! Profile generated!')
                    profileGenerated = True
                finally:
                    profileGenerated = True

            if profileGenerated == True:
                showActions = st.sidebar.checkbox("Select to show your app's operations manager:", value=False, key=12)
                if showActions == True:
                    st.sidebar.header("Data Operations Manager")

                    dataActions = st.sidebar.selectbox("Select a data operation:",
                                                       ("", "Data Profiling", "Data Wrangling"), index=0)

                    ######################
                    ### data profiling ###
                    ######################

                    if (dataActions == "Data Profiling"):
                        st.sidebar.header("✅ Data Profiling")
                        st.header("Data Profiling")

                        singleColumnAnalysisOptions = st.sidebar.radio("Single Column Analysis", (
                        "Cardinalities", "Value Distribution", "Data Types"), key=101)
                        if singleColumnAnalysisOptions == "Cardinalities":
                            st.subheader("Cardinalities")

                            ##########################################
                            ### cardinalities operations + metrics ###
                            ##########################################

                            missing_values_count = df.isnull().sum()
                            total_cells = np.product(df.shape)
                            n_cells_missing = report["table"]["n_cells_missing"]
                            total_missing = str(n_cells_missing) + " cells"
                            percent_numeric_missing = report["table"]["p_cells_missing"] * 100
                            percent_missing = "{:.2f}".format(percent_numeric_missing) + "%"
                            percent_completeness = "{:.2f}".format(100 - percent_numeric_missing) + "%"
                            total_completeness = total_cells - n_cells_missing
                            uniqueValuesInEachColumn = df.nunique()
                            uniqueValues = 0
                            for value in uniqueValuesInEachColumn:
                                uniqueValues += uniqueValuesInEachColumn[1]

                            percent_numeric_uniqueness = (uniqueValues / total_cells) * 100
                            percent_uniqueness = "{:.2f}".format(percent_numeric_uniqueness) + "%"

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(label="Number of rows", value=df.shape[0])
                            with col2:
                                st.metric(label="Number of attributes", value=df.size)
                            with col3:
                                st.metric(label="Completeness (number of none-null values)", value=total_completeness,
                                          delta=percent_completeness)
                            with col4:
                                st.metric(label="Uniqueness (number of distinct values)", value=uniqueValues,
                                          delta=percent_uniqueness)

                        elif singleColumnAnalysisOptions == "Value Distribution":
                            st.subheader("Value Distribution")
                            st.markdown(
                                "We are filtering the columns of your database including just numeric values, in order to analyze them!")
                            dfCol = []
                            for col in df.columns:
                                dfCol.append(col)
                            columnChosen = st.radio("Columns:", dfCol, key=102)
                            warningMessages = report["messages"]
                            warningToPrint = []
                            for mex in warningMessages:
                                checkWarning = mex[mex.find('[') + 1:mex.find(']')]
                                if columnChosen in mex:
                                    warningToPrint.append(checkWarning)
                            st.subheader("WARNINGS:")
                            if len(warningToPrint) != 0:
                                for warningMex in warningToPrint:
                                    st.info(
                                        "The column " + columnChosen + " chosen presents the following warning: " + str(
                                            warningMex))
                            else:
                                st.info("The column " + columnChosen + " chosen doesn't present any warnings!")
                            st.bar_chart(df[columnChosen])

                        elif singleColumnAnalysisOptions == "Data Types":
                            st.subheader("Data Types")
                            dfCol = []
                            for col in df.columns:
                                dfCol.append(col)
                            columnChosen = st.radio("Columns:", dfCol, key=103)

                            st.markdown(
                                "The column " + columnChosen + " chosen has type: " + str(df[columnChosen].dtype))

                    #####################
                    ### data wranging ###
                    #####################

                    if (dataActions == "Data Wrangling"):

                        ###############################################
                        ### creation of the wrangler-copy of the df ###
                        ###############################################

                        wrangler_df = df.copy(deep=True)
                        st.sidebar.header("✅ Data Wrangling")
                        st.header("Data Wrangling")
                        firstNewRows = st.checkbox("Select to show a preview of your wrangling dataset:", value=False,
                                                   key=13)

                        operationApllied = False

                        if firstNewRows == True:
                            st.write(wrangler_df.head())

                        wranglingOperations = ["Column Splitting", "Handling Missing Values", "Handling Dates",
                                               "Column Merging", "Column Renaming"]
                        activeOperations = []
                        count = 0
                        for op in wranglingOperations:
                            count += 1
                            activeOperations.append(st.sidebar.checkbox(op, value=False, key=count))
                            #activeOperations.append(st.sidebar.checkbox(op, value=False))

                        @st.cache
                        def convert_df(df):
                            return df.to_csv().encode('utf-8')

                        wrangler_df_csv = convert_df(wrangler_df)

                        for x in range(len(activeOperations)):
                            if (activeOperations[x] == True):

                                ########################
                                ### column splitting ###
                                ########################
                                if x == 0:
                                    st.subheader("Column Splitting")
                                    st.markdown(
                                        "This wrangling operation let you split a column composed of multiple values.")
                                    st.info(
                                        "👀Please select the column first, then the delimiter and then call your new columns whatever you want!")
                                    dfCol = []
                                    for col in wrangler_df.columns:
                                        if "," in str(wrangler_df[col][0]) or " " in str(wrangler_df[col][0]):
                                            dfCol.append(col)
                                    if (len(dfCol) > 0):
                                        columnChosen = st.radio("Columns with possible splitting:", dfCol, key=104)
                                        splittingDelimiter = st.selectbox("Select the delimiter of " + columnChosen,
                                                                          (" ", ","), 0)
                                        if splittingDelimiter == " ":
                                            if " " not in str(wrangler_df[columnChosen][0]):
                                                st.error(
                                                    "Sorry, there are no tuples in column " + columnChosen + " that agree with your pattern.\nPlease update column or delimiter!")
                                            else:
                                                st.success("Possibile wrangling!")

                                                firstColumn = st.text_input(
                                                    "Insert the name of the new column before the splitter: ")
                                                secondColumn = st.text_input(
                                                    "Insert the name of the new column after the splitter: ")
                                                split1 = wrangler_df[columnChosen].str.split(expand=True)

                                                applySplit = st.checkbox("Click here to apply", value=False, key=15)

                                                if applySplit == True:
                                                    wrangler_df[[firstColumn, secondColumn]] = wrangler_df[
                                                        columnChosen].str.split(splittingDelimiter, n=1, expand=True)
                                                    st.success(
                                                        "🎉Congrats!\nYou splitted column " + columnChosen + " into the two new columns: " + firstColumn + ", " + secondColumn)

                                                    deleteSplittedColumn = st.radio(
                                                        "Do you want to delete your splitted column " + columnChosen + "?",
                                                        ("Yes", "No"), 1, key=105)

                                                    if deleteSplittedColumn == "Yes":
                                                        wrangler_df = wrangler_df.drop(columnChosen, axis=1)
                                                        st.info("Column " + columnChosen + " has been removed!")
                                                    else:
                                                        st.info("Column " + columnChosen + " hasn't been removed!")

                                                    wrangler_df_csv = convert_df(wrangler_df)

                                                    showSplit = st.checkbox(
                                                        "Click here to show the new wrangled-dataset", value=False,
                                                        key=16)
                                                    if showSplit:
                                                        st.write(wrangler_df.head())

                                                    operationApllied = True

                                        if splittingDelimiter == ",":
                                            if "," not in str(wrangler_df[columnChosen][0]):
                                                st.error(
                                                    "Sorry, there are no tuples in column " + columnChosen + " that agree with your pattern.\nPlease update column or delimiter!")
                                            else:
                                                st.success("Possibile wrangling!")

                                                firstColumn = st.text_input(
                                                    "Insert the name of the new column before the splitter: ")
                                                secondColumn = st.text_input(
                                                    "Insert the name of the new column after the splitter: ")
                                                split1 = wrangler_df[columnChosen].str.split(expand=True)

                                                applySplit = st.checkbox("Click here to apply", value=False, key=17)
                                                if applySplit == True:
                                                    wrangler_df[[firstColumn, secondColumn]] = wrangler_df[
                                                        columnChosen].str.split(splittingDelimiter, n=1, expand=True)
                                                    st.success(
                                                        "🎉Congrats!\nYou splitted column " + columnChosen + " into the two new columns: " + firstColumn + ", " + secondColumn)

                                                    deleteSplittedColumn = st.radio(
                                                        "Do you want to delete your splitted column " + columnChosen + "?",
                                                        ("Yes", "No"), 1, key=106)
                                                    if deleteSplittedColumn == "Yes":
                                                        wrangler_df = wrangler_df.drop(columnChosen, axis=1)
                                                        st.info("Column " + columnChosen + " has been removed!")
                                                    else:
                                                        st.info("Column " + columnChosen + " hasn't been removed!")

                                                    wrangler_df_csv = convert_df(wrangler_df)
                                                    showSplit = st.checkbox("Click here to show your splitted dataset",
                                                                            value=False, key=18)
                                                    if showSplit:
                                                        st.write(wrangler_df.head())

                                                    operationApllied = True
                                    else:
                                        st.error(
                                            "Sorry, there are no tuples in your dataset that agree with your pattern. Please select another wrangling operation!")

                                ###############################
                                ### handling missing values ###
                                ###############################

                                if x == 1:
                                    st.subheader("Handling Missing Values")
                                    st.markdown(
                                        "This wrangling operation let you handle in different modes the missing values of your dataset.")
                                    writeColumns()

                                    ##########################################
                                    ### cardinalities operations + metrics ###
                                    ##########################################

                                    missing_values_count = df.isnull().sum()
                                    total_cells = np.product(df.shape)
                                    n_cells_missing = report["table"]["n_cells_missing"]
                                    total_missing = str(n_cells_missing) + " cells"
                                    percent_numeric_missing = report["table"]["p_cells_missing"] * 100
                                    total_percent_missing = "{:.2f}".format(percent_numeric_missing) + "%"

                                    st.markdown(
                                        "Your dataset has a total percentage of missing values of: " + total_percent_missing)

                                    if (n_cells_missing == 0):
                                        st.success("🎉Congrats! Your data set has no missing values")
                                    else:

                                        st.markdown(
                                            "You can handle missing values in 2 possible ways: by dropping or by filling!")
                                        dropOrReplace = st.selectbox("Select your handling mode:",
                                                                     ("", "Drop Data", "Fill Data"), index=0)

                                        if (dropOrReplace == "Drop Data"):
                                            st.markdown(
                                                "The dropping data mode is composed of other 2 options: full-drop mode and column-based-drop mode!")
                                            st.markdown("Choose the mode which best suits your necessities.")
                                            dropOption = st.selectbox("Check to show drop options:",
                                                                      ("", "Full-Drop Mode", "Column-Based-Drop Mode"),
                                                                      index=0)

                                            if (dropOption == "Full-Drop Mode"):
                                                wrangler_df = wrangler_df.dropna()
                                                st.markdown(
                                                    "Force the checkbox below if you want to delete every possible row that contains one or more missing values.")
                                                fulldrop = st.checkbox("Check to show the full-drop mode", key=19)
                                                if (fulldrop == True):
                                                    st.markdown("If you delete all the rows with missing values:")

                                                    col1, col2 = st.columns(2)
                                                    with col1:
                                                        st.markdown("Before wrangling dataset:")
                                                        st.dataframe(df)

                                                        st.metric(label="Number of rows", value=df.shape[0])
                                                        st.metric(label="Number of attributes", value=df.size)

                                                    with col2:
                                                        st.markdown("After wrangling dataset:")
                                                        st.dataframe(wrangler_df)

                                                        st.metric(label="Number of rows", value=wrangler_df.shape[0])
                                                        st.metric(label="Number of attributes", value=wrangler_df.size)

                                                    wrangler_df_csv = convert_df(wrangler_df)
                                                    operationApllied = True

                                            if (dropOption == "Column-Based-Drop Mode"):
                                                st.markdown(
                                                    "In this mode you can handle all the missing values for every colum that presents missing values.")
                                                st.markdown("Columns with missing values:")

                                                dfColToDrop = []
                                                dfCol = []
                                                for col in wrangler_df.columns:

                                                    if (wrangler_df[col].isnull().sum() != 0):
                                                        dfCol.append(col)
                                                        # dfColToDrop.append(st.checkbox(col, value=False, key=20))
                                                        dfColToDrop.append(st.checkbox(col, value=False))

                                                showStats = False

                                                for z in range(len(dfColToDrop)):

                                                    if dfColToDrop[z] == True:
                                                        showStats = True
                                                        columnCellMissing = report["variables"][dfCol[z]]["n_missing"]
                                                        percentCellMissing = report["variables"][dfCol[z]]["p_missing"]
                                                        realPercentCellMissing = "{:.2f}".format(
                                                            percentCellMissing) + "%"
                                                        st.markdown("Column " + dfCol[z] + " has a total of " + str(
                                                            columnCellMissing) + " missing values, which is " + str(
                                                            realPercentCellMissing) + ".")

                                                if showStats == True:
                                                    st.subheader("Changing stats:")
                                                    col1, col2 = st.columns(2)

                                                    with col1:
                                                        st.metric(
                                                            label="Before wrangling missing percentage",
                                                            value=total_percent_missing)
                                                    with col2:
                                                        afterMissingCount = wrangler_df.isnull().sum()

                                                        afterTotalMissing = afterMissingCount.sum()
                                                        afterMissingPercent = (afterTotalMissing / total_cells) * 100
                                                        afterTotalPercentMissing = "{:.2f}".format(
                                                            afterMissingPercent) + "%"

                                                        st.metric(label="After wrangling missing percentage",
                                                                  value=afterTotalPercentMissing)

                                                    missingWrangle = st.checkbox("Check to wrangle", value=False,
                                                                                 key=21)

                                                    if missingWrangle == True:

                                                        for w in range(len(dfColToDrop)):

                                                            if dfColToDrop[w] == True:
                                                                wrangler_df.drop(dfCol[w], axis=1, inplace=True)
                                                                st.info("❕Column " + dfCol[w] + " dropped!")

                                                        wrangler_df_csv = convert_df(wrangler_df)

                                                        newDF1 = st.checkbox(
                                                            "Select to show side by side comparison of your new-born wrangled dataset and the old one:",
                                                            value=False, key=22)

                                                        if newDF1 == True:
                                                            col1, col2 = st.columns(2)
                                                            with col1:
                                                                st.markdown("Before wrangling dataset:")
                                                                st.dataframe(df)
                                                            with col2:
                                                                st.markdown("After wrangling dataset:")
                                                                st.dataframe(wrangler_df)

                                                        operationApllied = True

                                        if (dropOrReplace == "Fill Data"):

                                            st.markdown(
                                                "Below you can find some possibilities of filling the missing values of your columns.")

                                            st.markdown("Columns with missing values:")

                                            dfCol = []
                                            dfColToFill = []

                                            for col in wrangler_df.columns:
                                                if (wrangler_df[col].isnull().sum() != 0):
                                                    dfCol.append(col)
                                                    #dfColToFill.append(st.checkbox(col, value=False, key=23))
                                                    dfColToFill.append(st.checkbox(col, value=False))

                                            oneFillingDone = False

                                            for y in range(len(dfColToFill)):

                                                if dfColToFill[y] == True:

                                                    if df[dfCol[y]].dtype == "object":
                                                        fillingOpObj = st.radio("Filling options for " + dfCol[y] + ":",
                                                                                (
                                                                                "", "Following Value", "Previous Value",
                                                                                "Mode", "Custom Value"), index=0,
                                                                                key=107)

                                                        if fillingOpObj == "Following Value":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:

                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        y] + " with its following value")
                                                                wrangler_df[dfCol[y]].fillna(method="bfill",
                                                                                             inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                                        if fillingOpObj == "Previous Value":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:

                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        y] + " with its previous value")
                                                                wrangler_df[dfCol[y]].fillna(method="ffill",
                                                                                             inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                                        if fillingOpObj == "Mode":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:
                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        x] + " with its mode")

                                                                wrangler_df[dfCol[y]].fillna(
                                                                    wrangler_df[dfCol[y]].mode()[0], inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                                        if fillingOpObj == "Custom Value":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:

                                                                customValue = st.text_input(
                                                                    "Please insert the custom value you want to use:")

                                                                if len(customValue) != 0:
                                                                    wrangler_df[dfCol[y]].replace([np.nan], customValue,
                                                                                                  inplace=True)
                                                                    st.info("✔️ Filled all missing values of column " +
                                                                            dfCol[y] + " with value " + customValue)

                                                                    wrangler_df_csv = convert_df(wrangler_df)
                                                                    oneFillingDone = True
                                                                    operationApllied = True



                                                    elif df[dfCol[y]].dtype == "float64":
                                                        fillingOpNum = st.radio("Filling options for " + dfCol[y] + ":",
                                                                                ("", "Max", "Avg", "0", "Mode"),
                                                                                index=0)#, key=108)

                                                        if fillingOpNum == "Max":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:

                                                                maxValue = report["variables"][dfCol[y]]["max"]
                                                                maxValue2 = "{:.2f}".format(maxValue)
                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        y] + " with its max value: " + maxValue2)
                                                                wrangler_df[dfCol[y]].replace([np.nan], maxValue,
                                                                                              inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True

                                                                operationApllied = True

                                                        if fillingOpNum == "Avg":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:
                                                                avgValue = report["variables"][dfCol[y]]["mean"]
                                                                avgValue2 = "{:.2f}".format(avgValue)
                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        y] + " with its average value: " + avgValue2)
                                                                wrangler_df[dfCol[y]].replace([np.nan], avgValue,
                                                                                              inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                                        if fillingOpNum == "0":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:
                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        y] + " with value 0")
                                                                wrangler_df[dfCol[y]].replace([np.nan], 0, inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                                        if fillingOpNum == "Mode":
                                                            if report["variables"][dfCol[y]]["n_missing"] == 0:
                                                                st.success(
                                                                    "Column " + dfCol[y] + " has no missing values!")
                                                            else:
                                                                st.info(
                                                                    "✔️ Filled all missing values of column " + dfCol[
                                                                        x] + " with its mode")

                                                                wrangler_df[dfCol[y]].fillna(
                                                                    wrangler_df[dfCol[y]].mode()[0], inplace=True)

                                                                wrangler_df_csv = convert_df(wrangler_df)
                                                                oneFillingDone = True
                                                                operationApllied = True

                                            if oneFillingDone == True:
                                                newDFFilled = st.checkbox(
                                                    "Check to show side by side comparison of your new-born wrangled dataset and the old one:",
                                                    value=False, key=24)
                                                if newDFFilled == True:
                                                    col1, col2 = st.columns(2)

                                                    with col1:
                                                        st.markdown("Before wrangling dataset:")
                                                        st.dataframe(df)
                                                    with col2:
                                                        st.markdown("After wrangling dataset:")
                                                        st.dataframe(wrangler_df)

                                                    wrangler_df_csv = convert_df(wrangler_df)

                                    ######################
                                    ### handling dates ###
                                    ######################

                                if x == 2:
                                    st.subheader("Handling Dates")
                                    st.markdown("Select the column that contains a date:")

                                    dfCol = []
                                    for col in wrangler_df.columns:
                                        dfCol.append(col)

                                    st.info("👀Please, make sure that your chosen column is a date!")
                                    st.info(
                                        "⚠️Remember that this operation, in case your column isn't an actual date, will catch an error! Be Careful!")

                                    columnHandlingChosen = st.radio("Columns:", dfCol, key=109)

                                    beforeWranglingDT = str(df[columnHandlingChosen].dtype)

                                    if df[columnHandlingChosen].dtype == "float64":
                                        st.markdown(
                                            "Column \"" + columnHandlingChosen + "\" has type: " + beforeWranglingDT)

                                        if df[columnHandlingChosen].dtype == "datetime64[ns]":
                                            st.success(
                                                "Column \"" + columnHandlingChosen + "\" has already the right datetime type!")
                                        else:

                                            st.error(
                                                "Invalid type of column \"" + columnHandlingChosen + "\".\nPlease change column!")

                                    elif df[columnHandlingChosen].dtype == "object":
                                        st.success("🎉Possible wrangling!")
                                        dateWrangle = st.checkbox(
                                            "Check to switch type of column " + columnHandlingChosen, value=False,
                                            key=25)

                                        if dateWrangle:
                                            try:
                                                wrangler_df[columnHandlingChosen] = pd.to_datetime(
                                                    wrangler_df[columnHandlingChosen], errors='coerce')

                                                showWrangleDate = st.button("Check to wrangle")

                                                if showWrangleDate == True:
                                                    col1, col2 = st.columns(2)

                                                    with col1:
                                                        st.metric(
                                                            "Before wrangling column " + columnHandlingChosen + " data type:",
                                                            beforeWranglingDT)
                                                    with col2:
                                                        st.metric(
                                                            "After wrangling column " + columnHandlingChosen + " data type:",
                                                            str(wrangler_df[columnHandlingChosen].dtype))

                                                    wrangler_df_csv = convert_df(wrangler_df)
                                                    operationApllied = True
                                            except NameError:
                                                st.error("Error while parsing column input. Please change column!")

                                ######################
                                ### column merging ###
                                ######################

                                if x == 3:
                                    st.subheader("Column Merging")

                                    dfColNoMiss = []
                                    for col in wrangler_df.columns:
                                        dfColNoMiss.append(col)

                                    st.info(
                                        "👀Remember: the first column chosen is the root-column of the mering operation!")

                                    firstMergingColumn = st.radio("Columns:", dfColNoMiss, key=110)

                                    dfColToMerge = []
                                    dfColMerge = []
                                    st.markdown("Columns with merging possibility:")
                                    count = 500
                                    for col in wrangler_df.columns:
                                        if (col != firstMergingColumn):
                                            count += 1
                                            dfColMerge.append(col)
                                            dfColToMerge.append(st.checkbox(col, value=False, key=count))#26))

                                    if len(dfColMerge) > 0:
                                        mergingPossibility = False

                                        for j in range(len(dfColToMerge)):

                                            if dfColToMerge[j] == True:
                                                mergingPossibility = True

                                        if mergingPossibility == True:
                                            mergecol1, mergecol2, mergecol3, mergecol4 = st.columns(4)

                                            with mergecol2:
                                                st.markdown("Merging column " + firstMergingColumn + " and columns: ")
                                            for h in range(len(dfColToMerge)):

                                                if dfColToMerge[h] == True:
                                                    operationApllied = True
                                                    with mergecol3:
                                                        st.markdown(dfColMerge[h])

                                                    wrangler_df[firstMergingColumn] = wrangler_df[
                                                                                          firstMergingColumn].map(
                                                        str) + "," + wrangler_df[dfColMerge[h]].map(str)
                                                    wrangler_df_csv = convert_df(wrangler_df)

                                            if operationApllied:
                                                st.info(
                                                    "💡Now that you merged some columns you can use the \"Column renaiming\" operation for column " + firstMergingColumn)

                                            showMerge = st.checkbox("Select to show your merged dataset", value=False,
                                                                    key=27)

                                            if showMerge == True:
                                                st.dataframe(wrangler_df)

                                #######################
                                ### column renaming ###
                                #######################

                                if x == 4:
                                    st.subheader("Column Renaming")

                                    dfColToRename = []
                                    dfColRename = []

                                    oneRename = False

                                    st.markdown("Select a column to show renaming operation:")
                                    count = 1000
                                    for columnIn in wrangler_df.columns:
                                        count += 1
                                        dfColRename.append(columnIn)
                                        dfColToRename.append(st.checkbox(columnIn, value=False, key=count))#28))

                                    renaming = False
                                    for q in range(len(dfColToRename)):

                                        if dfColToRename[q] == True:
                                            renaming = True

                                    if renaming == True:
                                        for q in range(len(dfColToRename)):

                                            if dfColToRename[q] == True:
                                                newName = st.text_input(
                                                    "Please insert a new name for column " + str(dfColRename[q]) + ":")
                                                if len(newName) > 0:
                                                    wrangler_df.rename(columns={dfColRename[q]: newName}, inplace=True)
                                                    wrangler_df_csv = wrangler_df.to_csv()

                                                    ########################################################################################################################
                                                    ### ATTENTION: change the folder path where you want to save the new wrangled-dataset with the column rename feature ###
                                                    ########################################################################################################################

                                                    wrangler_df.to_csv(r'NewDF.csv', index=False)

                                                    operationApllied = True

                                                    wrangler_df_csv = convert_df(wrangler_df)

                                                    oneRename = True

                                        if oneRename == True:
                                            showRenaming = st.checkbox("Select to show your column-renamed dataset",
                                                                       value=False, key=3333)

                                            if showRenaming:
                                                st.dataframe(wrangler_df)

                        if operationApllied == True:
                            st.sidebar.info("👇 \"Yes\" to download your wrangled-dataset! 👇")

                            showDlButton = st.sidebar.radio("❓Have you finished your operations?", ("No", "Yes"),
                                                            index=0, key=4444)

                            if showDlButton == "Yes":
                                showDownloadButton(wrangler_df_csv)

                else:
                    st.sidebar.info(
                        f"""
                        👆 Remember to check the "Profiling" checkbox!
                    """
                    )
            else:
                st.info(
                    f"""
                        👆 Generate the profiling report first in order to have accurate stats!
                    """
                )
    else:
        st.info(
            f"""
                👆 Upload a .csv file first!
            """
        )
