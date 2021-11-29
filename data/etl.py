import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dash
#import dash_core_components as dcc deprecated
from dash import dcc
from dash.dependencies import Input, Output
import mlflow.sklearn
#import dash_html_components as html deprecated
from dash import html
import plotly.express as px
import sys
from sqlalchemy import create_engine 



def load_data(filepath):
    """
    A function that loads and returns a dataframe.
    INPUT:
        filepath: relative or absolute path to the data
    RETURNS:
        df: Dataframe
    """
    df = pd.read_csv(filepath)
    return df

def business_understanding(dataframe):
    """
    A procedure to understand the business goal of the data.
    Prints the columns of the dataframe
    INPUT:
        dataframe: The data's dataframe to be understood
    RETURNS:
        None
    """
    columns = dataframe.columns
    print("The HR has identified and curated data from employees using nine criteria, namely:")
    for i, col in enumerate(columns):
        print(f"{i+1}. {col} ")
        
def data_cleaning_and_understanding(dataframe):
    """
    This procedure analyses the dataframe to answer the following questions:
        1. How many employees have left the company so far?
        2. Which group of employees have the tendency to leave most?
        3. Find other details about the data
        
        In particlar, this function analyzes the percentage change of each sub-group in every columns 
        between the original dataframe and the new dataframe involving employees that have left
        
        INPUT:
            dataframe- original dataframe
        OUTPUT:
            two series: dataframes with the column of the group of employees that are most likely to leave 
            their employments from both the original dataframe and the dataframe containing only those 
            that have left their employment
    """
    df = dataframe
    num_employees = df.shape[0]
    num_of_valid_employees = df.dropna(how="any").shape[0]
    num_of_missing_employees = df.shape[0] - num_of_valid_employees
    df_of_left_employees = df.query('LeaveOrNot==1')
    num_of_left_employees = df_of_left_employees.shape[0]
    # Get all the columns except the "LeaveOrNot"
    labels = df.columns[:-1]
    
    # Search for Group with the most tendency to leave their employment
    new = []
    original = []
    for label in labels:
        elem = df_of_left_employees[label].value_counts()/num_of_left_employees
        indiv = df[label].value_counts()/num_employees
        new.append(elem)
        original.append(indiv)
    percent_change = [(a-b)*100 for a, b in zip(np.array(new, dtype='object'), np.array(original, dtype='object'))]
    print('='*60)
    if num_of_missing_employees != 0:
        print(f"The dataset contains {num_of_valid_employees } valid employees.")
    else:
        print(f"The dataset contains {num_employees } number of employees.")
    print('='*60)
    
    print(f"{num_of_left_employees} employees have so far left the company")
    
    print('='*60)
    
    for x in percent_change:
        print(x)
        print('='*60)
    
    # This the group with the most tendency to leave their employment would be observed from the last printout above
    # The group with most percentage change is "Female" employees
    df_most_left_new = df_of_left_employees['Gender']
    df_most_left_original = df['Gender']
    
    return df_most_left_original, df_most_left_new
    

def create_dummy_df(df, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns that are categorical variables
            3. dummy columns for each of the categorical columns 
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    cat_df = df.select_dtypes(include=['object']) 
    #Create a copy of the dataframe


    #Pull a list of the column names of the categorical variables
    cat_cols = cat_df.columns
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue

    return df



def visualize_results(df):
    
    """
        A procedeure to plot the components of the column with the most tendency to leave before and after 
        those that left their employments
        
        INPUTS:
            df : Dataframe, df_before or df_after containing only the column of the group with the most tendency to 
            leave their employments
            
        RETURNS:
            None
            
    """
    df_left = df.query('LeaveOrNot==1')
    females_before = df['Gender'][df['Gender']=="Female"].count()
    males_before = df['Gender'][df['Gender']=="Male"].count()
    females_after = df_left['Gender'][df_left['Gender']=="Female"].count()
    males_after = df_left['Gender'][df_left['Gender']=="Male"].count()
    
    percent_of_females_before =   females_before*100/df.shape[0]
    percent_of_males_before   =   males_before*100/df.shape[0]
    
    percent_of_females_after  =   females_after*100/df_left.shape[0]
    percent_of_males_after    =   males_after*100/df_left.shape[0]
    X = ["Before_Resignation", "After_Resignation"]
    
    Females = [percent_of_females_before, percent_of_females_after]
    Males   = [percent_of_males_before, percent_of_males_after]
    
    X_axis = np.arange(len(X))
  
    plt.bar(X_axis - 0.2, Females, 0.4, label = 'Female Employees')
    plt.bar(X_axis + 0.2, Males, 0.4, label = 'Male Employees')
  
    plt.xticks(X_axis, X)
    plt.xlabel("Gender")
    plt.ylabel("Percentage of Employees by Gender")
    plt.title("Percentage of Employees in each group")
    plt.legend()
    plt.show()


def save_data(df, database_filename):
    """
    A procedure to save the cleaned dataframe returned from the create_dummy_df(df, dummy_na) function above in Sqlite database
    INPUTS:
        - df: -> returned dataframe
        - database_filename: -> The name of the table to save the df in the database
    RETURNS:
        NONE
        
    """
    # create an sqlite database engine using sqlalchemy
    engine = create_engine(f'sqlite:///{database_filename}')
    # save the df in a table if exists replace
    df.to_sql('ETLTable', engine, index=False, if_exists='replace')
    

def main():
    if len(sys.argv) == 3:

        filepath, database_filepath = sys.argv[1:]

        print('Loading data from: {}'.format(filepath))
        df = load_data(filepath)

        print('Presenting Business Understanding...')
        print('='*50)
        business_understanding(df)

        print("="*50)
        
        print("Presenting Data Understanding in Style")
        df_most_before, df_most_after = data_cleaning_and_understanding(df)
        print("="*50)
        
        print("Preparing Data For Machine Learning by Replacing Categorical Variables with Dummy Variables")
        df1 = create_dummy_df(df, dummy_na=False)
        print('='*50)

        
        
        print('Saving data ...\n  To   DATABASE: {}'.format(database_filepath))
        save_data(df1, database_filepath)
        
        print('Prepared data ready for building a model saved to database!')

        print('Visualing Data of the most likely group to resign')
        visualize_results(df)
        print('='*50)
    
    else:
        print('Please provide the filepaths of employee '\
              'dataset as the first as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the second argument. \n\nExample: python data/etl.py '\
              'data/Employee.csv'  'data/etlDatabase.db')



if __name__ == '__main__':
    main()