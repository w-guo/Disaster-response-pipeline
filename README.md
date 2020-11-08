# Disaster Response Pipeline 

### Project summary

In this project, we analyze a set of labelled disaster text messages which were collected from actual disaster events by Figure Eight, and build a machine learning pipeline using multiclass logistic regression that is able to categorize emergency messages based on the needs. We also set up a Flask web application that allows an emergency worker to input a new message and get classified results in several categories so that messages can be directed to an appropriate disaster relief agency.

### File descriptions

    ├── app     
    │   ├── run.py                           # Flask file that runs app
    │   └── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    ├── data                   
    │   ├── disaster_categories.csv          # Dataset including all the categories  
    │   ├── disaster_messages.csv            # Dataset including all the messages
    │   └── process_data.py                  # Build a data cleaning pipeline
    ├── models
    │   └── train_classifier.py              # Build a machine learning pipeline           
    └── README.md

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
