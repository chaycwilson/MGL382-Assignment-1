import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = {
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Male'],
    'Married': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Dependents': [0, 1, 1, 0, 1],
    'Education': ['Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate'],
    'Self_Employed': ['No', 'No', 'Yes', 'No', 'No'],
    'ApplicantIncome': [5849, 4583, 3000, 2583, 6000],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'Y']
}
df = pd.DataFrame(data)

# Train the model
X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome']]
y = df['Loan_Status']
model = RandomForestRegressor()
model.fit(X, y)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Loan Prediction"),
    html.Label("Gender: "),
    dcc.Input(id='gender', type='text', value=50),
    html.Label("Married: "),
    dcc.Input(id='married', type='text', value=-34.6),
    html.Label("Dependents: "),
    dcc.Input(id='dependents', type='number', value=-58.4),
    html.Label("Education:"),
    dcc.Input(id='education', type='text', value=''),
    html.Label('Self_Employed:'),
    dcc.Input(id='self_employed', type='text'),
    html.Label('ApplicantIncome:'),
    dcc.Input(id='applicant_income', type='number'),
    html.Button('Predict', id='submit-val', n_clicks=0),
    html.Div(id='output-prediction')
])

# Define callback to update prediction
@app.callback(
    Output('output-prediction', 'children'),
    [Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('gender', 'value'),
    dash.dependencies.State('married', 'value'),
    dash.dependencies.State('dependents', 'value'),
    dash.dependencies.State('education', 'value'),
    dash.dependencies.State('self_employed', 'value'),
    dash.dependencies.State('applicant_income', 'value')])
def update_output(n_clicks, gender, married, dependents, education, self_employed, applicant_income):
    prediction = model.predict([[gender, married, dependents, education, self_employed, applicant_income]])
    return f"The predicted house price is ${prediction[0]:,.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)