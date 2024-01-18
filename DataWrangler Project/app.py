import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

####################
### IMPORT PAGES ### 
####################

from multipage import MultiPage
from pages import presentation_page, automatic_data_wrangling, data_uploading
from pages.data_uploading import uploading_csv

# Create an instance of the app 
app = MultiPage()

st.set_page_config(page_title="DataWrangler", layout="wide")


##################
### IMPORT CSS ###
##################

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



#################
### ADD PAGES ###
#################

app.add_page("Presentation Page", presentation_page.app)
app.add_page("Data Wrangling Page", automatic_data_wrangling.app)


############
### MAIN ###
############

app.run()
