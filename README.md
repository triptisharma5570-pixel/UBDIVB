# Universal Bank — Personal Loan Streamlit App

This repository contains a Streamlit app (`app.py`) that:
- Loads the Universal Bank dataset (sample included)
- Provides 5 advanced EDA charts for marketing insights
- Trains Decision Tree, Random Forest, and Gradient Boosting models with a single click
- Produces performance metrics (accuracy, precision, recall, F1, AUC) and combined ROC
- Allows uploading a new CSV to generate predictions and download the labelled file

## How to deploy on Streamlit Cloud
1. Create a new GitHub repository and push these files (no folders). 
2. In Streamlit Cloud, create a new app and point it to `app.py` in your repo.
3. Add `requirements.txt` packages (already included). Streamlit Cloud will install them.

## Notes
- This project uses default package names in `requirements.txt` (no pinned versions) as requested.
- Models are kept in Streamlit session state (they are not persisted across app restarts). To persist models for production, save them to cloud storage or add a model artifact to the repo.


