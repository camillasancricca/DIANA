import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Tool name", layout="wide", initial_sidebar_state="collapsed")

st.title("Welcome to the introduction page")

start_button = st.button("Let's start the analysis")
st.session_state['x'] = 0
if start_button:
    switch_page("upload")
