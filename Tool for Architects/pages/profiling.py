import streamlit as st
import pandas as pd
import numpy as np
#import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from streamlit_extras.switch_page_button import switch_page


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(255, 254, 239);
    height:auto;
    width:auto;
}
</style>""", unsafe_allow_html=True)

df = st.session_state['df']
if st.session_state['Once'] == True:
    pr = df.profile_report()
    st_profile_report(pr)
    st.session_state['Once'] = False

if st.button("Back to homepage"):
        switch_page("Homepage")

