import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt', 'stopwords', 'wordnet'])

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

def load_data(database_filepath):
    """Load data from database

    Args:
    database_filepath: file path where database was saved

    Returns: 
    X: feature variables
    Y: target variables
    category_names: names of categories in Y
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    """Preprocess text data

    Args: 
    text: raw text data

    Returns: 
    tokens: list of strings after tokenization
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(word)
              for word in tokens if word not in stopwords.words("english")]

    return tokens


def build_model():
    """Create a machine learning pipeline

    Returns:
    pipeline: machine learning pipeline model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(random_state=2020), n_jobs=-1))
    ])

    parameters = {
        'clf__estimator__C': [1, 2, 4],
        'clf__estimator__penalty': ['l1', 'l2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate training model on test set

    Args: 
    model: training model
    X_test: samples in test set 
    Y_test: labels in test set 
    category_names: names of categories in Y_test
    """
    Y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        y_true = Y_test.values[:, i]
        y_pred = Y_pred[:, i]
        target_names = ['not {}'.format(column), '{}'.format(column)]
        print(classification_report(
            y_true, y_pred, target_names=target_names))


def save_model(model, model_filepath):
    """Save model as a pickle file

    Args:
    model: training model to be saved
    model_filepath: file path where model is saved
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
