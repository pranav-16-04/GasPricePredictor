from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import datetime
import os
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)

joblib_path = os.path.join(os.path.dirname(__file__), 'gas.joblib')

try:
    data = joblib.load(joblib_path)
    model = data['model']
    scaler = data['scaler']
except Exception as e:
    app.logger.error(f"Error loading model and scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Model and scaler are not loaded correctly.')

    try:
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])

        date = datetime.datetime(year, month, day)
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        
        input_features = pd.DataFrame([[year, month, day, day_of_week, is_weekend]], 
                                      columns=['year', 'month', 'day', 'day_of_week', 'is_weekend'])

       
        input_features_scaled = scaler.transform(input_features)

        
        prediction = model.predict(input_features_scaled)[0]

        
        prediction = max(0, prediction)

        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}')
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
