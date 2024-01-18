# Data Wrangler App

## 📥 Install the environment
To install the streamlit environment please follow this quick installation guide:
https://docs.streamlit.io/library/get-started/installation

## 🛎️ Requirements
You can find all the libraries used in this project in the requirements.txt.

## 🖥️ Deploy local app
Clone this app, simply by downloading the zip file of the repo. Open it on your local pc.

⚠️ Remember to change inside the "automatic_data_wrangling.py" script:
- folder path where the json profile report will be saved (line 59)
- folder path where the wrangled-dataset with column rename feature will be saved (line 702)

## 🏃 Run
👀 Everytime you change dataset remember to delete the old json report! 👀

From your conda environment:

streamlit run app.py

or, to avoid 200MB upload file limit:

streamlit run app.py --server.maxUploadSize=1028
