import dash
#import dash_core_components as dcc
from dash import dcc
from dash.dependencies import Input, Output
#import dash_html_components as html
from dash import html
import plotly.express as px
import joblib
import pandas as pd



# Load the model externally
model = joblib.load('./model/model.pkl')

# Define a dash app

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(children=[
    html.H1(children='Model UI'),
    html.H2(children='Prediction Whether an Employee would Resign or Not'),
    html.H3(children='Prediction value of 0 means the employee would (did) not resign'),
    html.H4(children='PaymentTier: (1, 2 or 3) ::: Age: (20 - 60) :::  ExperienceInCurrentDomain: (0, 1, 2, 3, 4, 5, 6 or 7)'),
    
    html.Div([
        html.Label('JoiningYear'),
        dcc.Input(value='2017', type ='number', id='yr'),
    ]),
    
    html.Div([
        html.Label('PaymentTier:'),
        dcc.Input(value='2', type ='number', id='level'),
    ]),
    html.Div([
        html.Label('Age:'),
        dcc.Input(value='25', type ='number', id='age'),
    ]),
    html.Div([
        html.Label('ExperienceInCurrentDomain:'),
        dcc.Input(value='3', type ='number', id='exp'),
    ]),
    html.P([
        html.Label('Prediction'),
        dcc.Input(value='0', type ='text', id='pred'),
    ]),
])


# Callback

@app.callback(

    Output(component_id="pred", component_property="value"),
    [
     Input(component_id="yr", component_property="value"),
     Input(component_id="level", component_property="value"),
     Input(component_id="age", component_property="value"),
     Input(component_id="exp", component_property="value")
    
    ]
)



def update_prediction( year, level,age, exp):
    new_row = {
        
        "JoiningYear": year,
        "PaymentTier": level,
        "Age": age,
        "ExperienceInCurrentDomain": exp,
        "Education_Masters" : 1,
        "Education_PHD":  0,
        "City_New Delhi": 0,
        "City_Pune": 1 ,
        "Gender_Male":  1 ,
        "EverBenched_Yes": 1
    }
    
    new_df = pd.DataFrame.from_dict(new_row, orient="index").transpose()
    
    
    # predict the output of the new_df
    return str(model.predict(new_df)[0])
    
    


if __name__=='__main__':
    app.run_server(debug=True)