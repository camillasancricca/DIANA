import os
import streamlit as st
import time


def app():
    st.title("Data Wrangling App")
    
    st.markdown("This project is based on the thesis: 'Context Aware Data Preparation', by Camilla Sancricca and supervised by prof. Cinzia Cappiello.")
    st.markdown("This tool is developed in order to help and support the user during their data wrangling operations.")
    st.subheader("Walkthrough")
    st.markdown("In the sidebar you can find a navigation bar which is composed of two main pages: the presentation page (this one) and the wrangling app.")
    st.markdown("By clicking on the wrangling app an uploader file shows up and you can upload the .csv file you want to work on.")
    st.markdown("Once the uploading is confirmed the tool let you create the profiling of your dataset, which is important to get informations and stats before and after every wrangling operation.")
    st.markdown("At this point if you click on \"Data profiling\" you can see the insights of your dataset.")
    st.markdown("If you click on \"Data Wrangling\" you can select between the following wrangling operations:")
    st.markdown("- ✂️**Splitting**: you can split a column into two different columns, nameing the two new columns with also the possibility to delete the column splitted")
    st.markdown("- 🗑️ **Handling Missing Values**: you can drop or fill the tuples with missing values")
    st.markdown("- 📆 **Handling Dates**: you can change the type of a date from object to datetime")
    st.markdown("- 🫂 **Merging**: you can merge two or more columns")
    st.markdown("- 💱 **Renaming**: you can rename one or more columns")
    st.info("You can operate one or more wrangling operations together. Remember to let the tool know (from the sidebar) that you finished your operations in order to generate the download .csv button.")
    st.error("⚠️REMEMBER: when you are one the wrangling page do NOT call back the data profiling page. You will lost every operation done!")
    
