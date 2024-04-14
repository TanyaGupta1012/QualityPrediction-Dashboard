from flask import Flask, render_template, request
import numpy as np
import pickle
import wine
import air
import pandas as pd

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    water_model = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model_air = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/water_predict', methods=['POST', 'GET'])
def water_predict():
    # Extract input data from the form
    input_data = [float(x) for x in request.form.values()]
    input_data_reshape = np.array(input_data).reshape(1, -1)

    # Make prediction using the pre-trained model
    prediction = water_model.predict(input_data_reshape)
    # prediction = model.predict([input_data])


    # Determine prediction result
    if prediction[0] == 0:
        result = "Level of water is good"
    else:
        result = "Level of water is polluted, alert system is activated"

    # Return prediction result to the user
    return render_template('result.html', result=result)


@app.route('/wine_predict', methods=['POST'])
def wine_predict():
    # Extract input data from the form
    input_data = [float(x) for x in request.form.values()]

    # Make prediction using the function from wine.py
    prediction = wine.predict_wine_quality(input_data)

    # Return prediction result to the user
    return render_template('result.html', prediction=prediction)


# with open('model.pkl', 'rb') as f:
#     model_air = pickle.load(f)


# Assuming df2 is the DataFrame containing your preprocessed data
@app.route('/air_predict', methods=['POST', 'GET'])
def air_predict():
    # print("Request received for air prediction")
    input_data = [float(x) for x in request.form.values()]
    # print("Input data:", input_data)
    predicted_aqi = air.predict_aqi(model_air, input_data)
    # print("Predicted AQI:", predicted_aqi)
    return render_template('result.html', air_prediction=predicted_aqi)

if __name__ == '__main__':
    app.run(debug=True)