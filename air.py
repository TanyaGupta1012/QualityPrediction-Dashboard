import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression_model(df):
    # Discretize the continuous target variable into classes
    n_bins = 5  # Number of bins for discretization
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    y_discrete = discretizer.fit_transform(df["AQI"].values.reshape(-1, 1)).flatten()

    # Splitting data into features and target variable
    x = df.drop("AQI", axis=1)
    y = pd.Series(y_discrete)

    # Splitting into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Oversample using SMOTE
    oversample = SMOTE(k_neighbors=4)
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

    # Initialize and train the Logistic Regression model
    model_lr = LogisticRegression()
    model_lr.fit(x_resampled, y_resampled)

    return model_lr

def predict_aqi(model, input_data):
    # Input data for prediction
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshape = input_data_as_np_array.reshape(1, -1)

    # Prediction on input data
    prediction = model.predict(input_data_reshape)
    return prediction[0]
