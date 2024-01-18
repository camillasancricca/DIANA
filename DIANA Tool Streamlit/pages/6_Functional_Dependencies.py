from cgitb import small
import os
import webbrowser

import navigation as navigation
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

fd_data = st.session_state.setdefault("functional_dependencies",{})
df = st.session_state['df']
if 'my_dataframe' not in st.session_state:
    st.session_state.my_dataframe = df

automatic_fds = st.session_state.setdefault("functional_dependencies",{})



st.title("Functional Dependencies")

with st.form(key='new_fd_form'):
    left_side= st.multiselect("Left Side",st.session_state.my_dataframe.columns)
    right_side = st.multiselect("Right Side",st.session_state.my_dataframe.columns)
    if st.form_submit_button("Add FD"):
        if left_side and right_side:
            fd_name = f"{left_side} -> {right_side}"
            if fd_name not in automatic_fds:
                automatic_fds[fd_name] = {'left_side': left_side, 'right_side': right_side}
                fd_refl = f"{right_side} -> {right_side}"
                fd_reflo = f"{left_side} -> {left_side}"
                if fd_refl not in automatic_fds:
                    automatic_fds[fd_refl] = {'left_side': right_side, 'right_side': right_side}
                if fd_reflo not in automatic_fds:
                    automatic_fds[fd_reflo] ={'left_side': left_side, 'right_side': left_side}
                for other_col in st.session_state.my_dataframe.columns:
                    fd_trans = f"{right_side} -> ['{other_col}']"
                    if fd_trans in automatic_fds:
                        fd_transitive = f"{left_side} -> ['{other_col}']"
                        if fd_transitive not in automatic_fds:
                            automatic_fds[fd_transitive] = {'left_side': left_side, 'right_side': f"['{other_col}']"}

            else:
                st.warning("This FD already exists. Please enter a new FD.")



st.header("Existing FDa")
if fd_data:
    st.write(automatic_fds)
else:
    st.info("No FDs have been added yet")

fd_to_remove = st.selectbox("Select an FD to remove", list(fd_data.keys()))
if st.button("Remove FD"):
    if fd_to_remove in automatic_fds:
        del automatic_fds[fd_to_remove]
        st.success("FD removed successfully.")
    else:
        st.warning("FD not found.")

st.write("---")

if button("Change dataset", key="changedataset"):
    if 'my_dataframe' not in st.session_state:
        st.session_state.my_dataframe = df
    st.warning("If you don't have the modified dataset downloaded, you'll lose all the changes applied.")
    if st.button("Proceed"):
        st.session_state['x'] = 0
        switch_page("upload")

if st.button("Continue", key="continue_cleaning"):
        switch_page("Cleaning")


if st.button("Come Back", key="come_back_profiling"):
    switch_page("Categorical_Dependencies")
