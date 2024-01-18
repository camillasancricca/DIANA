import pandas as pd
import numpy as np
import streamlit as st

##################
### UPLOAD CSV ###
##################

def uploading_csv():
    st.header("📁Upload your dataset:")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
