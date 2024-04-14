import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("water_potability (2).csv")

# Handle missing values
df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

# Prepare features and target
x = df.drop("Potability", axis=1)
y = df["Potability"]

# Standardize features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

# Make predictions
pred_lr = model_lr.predict(x_test)
accuracy_score_lr = accuracy_score(y_test, pred_lr)

# # Define input data for prediction
input_data = np.array([value1, value2, value3, ..., valueN])

# # Reshape input data and make prediction
input_data_reshape = input_data.reshape(1, -1)
prediction = model_lr.predict(input_data_reshape)

with open('model.pkl', 'wb') as f:
    pickle.dump(model_lr, f)


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pickle

# def train_and_save_model(dataset_file):
#     # Load dataset
#     df = pd.read_csv(dataset_file)

#     # Handle missing values
#     df["ph"] = df["ph"].fillna(df["ph"].mean())
#     df["Sulfate"] = df["Sulfate"].fillna(df["Sulfate"].mean())
#     df["Trihalomethanes"] = df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())

#     # Prepare features and target
#     x = df.drop("Potability", axis=1)
#     y = df["Potability"]

#     # Standardize features
#     scaler = StandardScaler()
#     x = scaler.fit_transform(x)

#     # Split dataset into train and test sets
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#     # Logistic Regression model
#     model_lr = LogisticRegression()
#     model_lr.fit(x_train, y_train)

#     # Make predictions
#     pred_lr = model_lr.predict(x_test)
#     accuracy_score_lr = accuracy_score(y_test, pred_lr)

#     # Save the trained model
#     with open('model.pkl', 'wb') as f:
#         pickle.dump(model_lr, f)

# def predict_water_potability(input_data):
#     # Load the trained model
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)

#     # Reshape input data and make prediction
#     input_data_reshape = np.array(input_data).reshape(1, -1)
#     prediction = model.predict(input_data_reshape)

#     return prediction
