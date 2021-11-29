import pandas as pd
import numpy as np
from sqlalchemy import create_engine 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import mlflow.sklearn
import sys
import pickle


def load_data(database_filepath):
    """
    A procedure to load dataframe from a sqlite database and the dataframe
    
    INPUT/ARG:
        database_filepath-> database filepath
    OUTPUTS/RETURNS
        df: The read dataframe
    
        
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    conn = engine.connect()

    df = pd.read_sql_table('ETLTable', con=conn)
    return df


def pipeline(df):
    """
    A function to train a model using XGBoostClassifier.
    
    INPUT:
        df - data_frame with dummy variables in place of categorical variables
    OUTPUTS:
        X_test - X_test  
        y_test - y_test
        model  - trained model
    """
    X = df.drop('LeaveOrNot', axis=1)
    Y = df['LeaveOrNot']
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)    
    
    # Define the model and train or fit it
    
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.23987339304320798,
        n_estimators=113,
        gamma=0.880334867665456,
        eval_metric="error",
        use_label_encoder=False
        ).fit(X_train,y_train)
        
    return X_test, y_test, model

def evaluate_model(X_test, y_test, model):
    """
    Evaluates the model by displaying the precision, recall, f1_score, and accuracy of the model
    INPUTS:
        X_test, y_test, model
    OUTPUTS:
        None
    
    """
    y_true = model.predict(X_test)
    print('='*50)
    print(classification_report(y_test, y_true))
    print('='*50)
    print(f"The accuracy of the model is:  {accuracy_score(y_test, y_true)*100}")


def save_model(model, model_filepath):
    # Export and save the model to the model_filepath a pickle file
    pickle.dump(model, open(model_filepath, "wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)
        
        
        print('Building model...')
        X_test, y_test, model = pipeline(df)
        
        print('='*50)
        print()
        
        
        print('Evaluating model...')
        evaluate_model(X_test, y_test, model)
        print('='*50)


        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('='*50)
        print()

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the employee database '\
              'as the first argument and the filepath of the mlflow file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'model.py data/etlDatabase.db  model/')


if __name__ == '__main__':
    main()