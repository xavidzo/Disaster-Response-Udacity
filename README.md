# Disaster Response Pipeline Project

### Description
The goal of this project is to realize two pipelines, namely one ETL (Extract, Transform, Load) pipeline that processes the raw .csv files containing disaster messages and disaster categories (provided by former company Figure 8) and creates a new database with cleaned and combined entries. The second one is a ML pipeline that makes use of the previous database and by means of NLP (Natural Language Processing) techniques transforms the data into a format suitable for a classifier to predict the most likely event categories for the disaster messages. Moreover, a flask app let us visualize in the web browser some information about the dataset and the prediction results, so the application has an interactive character with the user in real-time.

### Dependencies
* Python 3
* NumPy, SciPy, Pandas, scikit-learn
* NLTK
* SQLalchemy
* Pickle
* Flask, Plotly
* jupyter

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/


### Files structure:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification results page of web app
|- run.py  # Script that runs the app

- data
|- disaster_categories.csv  # data with disaster event categories
|- disaster_messages.csv  # data with messages collected in several disaster situations
|- process_data.py # script that executes the ETL pipeline
|- DisasterResponse.db   # database where the cleaned data is saved into

- models
|- train_classifier.py # script that executes the ML pipeline
|- classifier.pkl  # saved model as pickle file

- notebooks
|- ETL_Pipeline_Preparation.ipynb # jupyter notebook that served to complete process_data.py
|- ML_Pipeline_Preparation.ipynb # jupyter notebook that served to complete train_classifier.py

- README.md
```

