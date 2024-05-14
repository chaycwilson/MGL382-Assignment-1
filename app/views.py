from app import app
from flask import render_template, request
import pandas as pd
from sklearn.externals import joblib

# You would load your model here
model = joblib.load('path_to_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assume you receive data from a form, you process it and predict
        data = request.form.to_dict()
        dataframe = pd.DataFrame(data, index=[0])
        prediction = model.predict(dataframe)
        return render_template('result.html', prediction=prediction)

    return render_template('index.html')
