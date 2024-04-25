import pandas as pd
import pickle

# Read the dataset
df = pd.read_csv('winequality.csv')

# Extract feature names
feature_names = df.columns.tolist()[:-1]  # Exclude the target column (quality)

# Save feature names to a pickle file
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
