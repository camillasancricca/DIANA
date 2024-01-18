# Lorenzo Vela Thesis

## Install the environment
To install the streamlit environment in Anaconda:
pip install streamlit==version

The current streamlit version used by the app is 1.15.1

Additional packages to be downloaded:
- pip install streamlit-extras
- pip install streamlit-nested-layout
- pip install streamlit-pandas-profiling

## Requirements
You can find all the libraries used in this project in the requirements.txt.

## Deploy the app
Clone this app, simply by downloading the zip file of the repo. Open it on your local pc.

## Run

From your conda environment:

streamlit run app.py

or, to avoid 200MB upload file limit:

streamlit run app.py --server.maxUploadSize=1028
