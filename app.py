import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Use classifier for categorical target

# Load and prepare the data
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

# One-hot encode categorical variables
X = pd.get_dummies(df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome']], drop_first=True)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Encoding 'Loan_Status' as 0 and 1

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Loan Prediction"),
    html.Label("Gender (Male or Female): "),
    dcc.Input(id='gender', type='text', value='Male'),
    html.Label("Married (Yes or No): "),
    dcc.Input(id='married', type='text', value='Yes'),
    html.Label("Dependents (Number of dependents): "),
    dcc.Input(id='dependents', type='number', value=0),
    html.Label("Education (Graduate or Not Graduate):"),
    dcc.Input(id='education', type='text', value='Graduate'),
    html.Label('Self Employed (Yes or No):'),
    dcc.Input(id='self_employed', type='text', value='No'),
    html.Label('Applicant Income (Numeric):'),
    dcc.Input(id='applicant_income', type='number', value=5000),
    html.Button('Predict', id='submit-val', n_clicks=0),
    html.Div(id='output-prediction')
])

# Define callback to update prediction
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
    try:
        # Preparing input DataFrame based on input
        input_df = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [int(dependents)],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [int(applicant_income)]
        })

        # Creating dummy variables
        input_df = pd.get_dummies(input_df)

        # Define the expected columns as per the training dataset
        expected_columns = ['Gender_Male', 'Gender_Female', 'Married_Yes', 'Married_No', 
                            'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_Yes', 
                            'Self_Employed_No', 'ApplicantIncome', 'Dependents']

        # Add missing dummy columns as 0
        for column in expected_columns:
            if column not in input_df.columns:
                input_df[column] = 0

        # Reorder columns as per the training model
        input_df = input_df[expected_columns]

        # Predict using the model
        prediction = model.predict(input_df)
        status = 'Approved' if prediction >= 0.5 else 'Not Approved'
        return f"The loan status is {status}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
