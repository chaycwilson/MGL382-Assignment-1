import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Data Preparation
data = {
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Male'],
    'Married': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Dependents': [0, 1, 1, 0, 1],
    'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
    'Self_Employed': ['No', 'No', 'Yes', 'No', 'No'],
    'ApplicantIncome': [5849, 4583, 3000, 2583, 6000],
    'Loan_Status': ['Y', 'N', 'N', 'N', 'Y']
}
df = pd.DataFrame(data)

# Feature Engineering
X = pd.get_dummies(df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome']], drop_first=True)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Model Training
model = RandomForestClassifier()
model.fit(X, y)

# Dash App Initialization
# Dash App Initialization
app = dash.Dash(__name__)

# CSS Styles
styles = {
    'container': {
        'margin': '50px',
        'padding': '20px',
        'border': '1px solid #ddd',
        'border-radius': '5px',
        'background-color': '#f9f9f9'
    },
    'label': {
        'margin': '10px 0 5px'
    },
    'input': {
        'width': '100%',
        'padding': '10px',
        'margin': '10px 0'
    },
    'button': {
        'width': '100%',
        'padding': '10px',
        'background-color': '#007BFF',
        'color': 'white',
        'border': 'none',
        'border-radius': '5px',
        'cursor': 'pointer'
    },
    'output': {
        'marginTop': '20px',
        'padding': '10px',
        'background-color': '#e2f0cb',  # Light green for positive output
        'border': '1px solid #d4d4d4',
        'border-radius': '5px',
        'color': '#38761d',  # Dark green text
        'font-size': '16px',
        'text-align': 'center'
    }
}

# App Layout
app.layout = html.Div(style=styles['container'], children=[
    html.H1("Loan Prediction"),
    html.Label("Gender (Male or Female):", style=styles['label']),
    dcc.Input(id='gender', type='text', style=styles['input']),
    html.Label("Married (Yes or No):", style=styles['label']),
    dcc.Input(id='married', type='text', style=styles['input']),
    html.Label("Dependents (Number of dependents):", style=styles['label']),
    dcc.Input(id='dependents', type='number', style=styles['input']),
    html.Label("Education (Graduate or Not Graduate):", style=styles['label']),
    dcc.Input(id='education', type='text', style=styles['input']),
    html.Label("Self Employed (Yes or No):", style=styles['label']),
    dcc.Input(id='self_employed', type='text', style=styles['input']),
    html.Label("Applicant Income (Numeric):", style=styles['label']),
    dcc.Input(id='applicant_income', type='number', style=styles['input']),
    html.Button('Predict', id='submit-val', n_clicks=0, style=styles['button']),
    html.Div(id='output-prediction', style=styles['output'])
])

# Callback for updating the prediction
@app.callback(
    Output('output-prediction', 'children'),
    [Input('submit-val', 'n_clicks')],
    [
        dash.dependencies.State('gender', 'value'),
        dash.dependencies.State('married', 'value'),
        dash.dependencies.State('dependents', 'value'),
        dash.dependencies.State('education', 'value'),
        dash.dependencies.State('self_employed', 'value'),
        dash.dependencies.State('applicant_income', 'value')
    ])
def update_output(n_clicks, gender, married, dependents, education, self_employed, applicant_income):
    try:
        input_df = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [int(dependents)],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [int(applicant_income)]
        })

        # Handling categorical variables
        input_df = pd.get_dummies(input_df)
        expected_columns = ['Dependents', 'ApplicantIncome', 'Gender_Male', 'Married_Yes', 
                            'Education_Not Graduate', 'Self_Employed_Yes']
        
        # Ensuring all required dummy variables are present
        missing_cols = set(expected_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[expected_columns]

        # Prediction
        prediction = model.predict(input_df)
        status = 'Approved' if prediction[0] == 1 else 'Not Approved'
        return f"The loan status is {status}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
        

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

