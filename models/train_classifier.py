import sys
# import libraries
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import re

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score




def load_data(database_filepath):
    """Loads data from database to deliver the necessary variables for the classification algorithm
    Args:
        database_filepath (string): the file path pointing to the database
    Returns:
        X, Y, category_names (tuple): X is the input variable, Y the labels, and category_names
        correspond to the columns of Y.
    """

    # data from previous table stored in database file- is loaded into a pandas df
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_response", engine)

    # X and Y are defined as input and label variables respectively
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """This represents the tokenize function used later in the pipeline
    Args:
        text (string): sentence or paragraph to be processed
    Returns:
        tokens (list of strings): individual elements of input text after tokenization and lemmatization
    """

    # text is converted to lowercase, separated in tokens, i.e. words, and each token is lemmatized
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())   
    tokens = word_tokenize(text)   
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]    
    
    return tokens


def build_model():
    """Function creates the pipeline to transform and classify the data
    Args:
        None
    Returns:
        cv (GridSearchCV object): it contains the hyperparameters to optimize the pipeline later
    """

    # Pipeline tranforms input data into tokens, then is mapped to TFIDF,
    # which is fed into a Random Forest for final classification
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameters to try in different combinations in Grid Search to find the optimal setting
    parameters = {
        'clf__estimator__n_estimators': [40, 60, 80],
        'clf__estimator__min_samples_split': [4, 6, 8],    
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5, verbose=2)
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """Function computes and prints the accuracy, precision and recall metrics for every category
    Args:
        model (sklearn GridSearchCV object): model after optimization
        X_test (pandas df): input data for testing
        Y_test (pandas df): labels of X_test
        category_names (list of strings): names for each column in Y_test
    Returns:
        None
    """
    Y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test):
        print(col,':')
        print('Accuracy: ', accuracy_score(Y_test.iloc[:,i], Y_pred[:,i]))
        print('Precision: ', precision_score(Y_test.iloc[:,i], Y_pred[:,i], average='weighted'))
        print('Recall: ', recall_score(Y_test.iloc[:,i], Y_pred[:,i], average='weighted'))
        print('--------------------------------')

def save_model(model, model_filepath):
    """Saves the previous trained model as a pickle file in the specified location
    Args:
        model (sklearn GridSearchCV object): model after optimization
        model_filepath (string): the path pointing to the pickle output file
    Returns:
        None
    """

    # The trained model is saved into a serialized pickle file
    with open(model_filepath, 'wb') as pklfile:
        pickle.dump(model, pklfile)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()