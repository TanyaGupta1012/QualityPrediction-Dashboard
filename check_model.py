import pickle

try:
    # Load the content of model.pkl
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    # Print the content
    print("Model data:", model_data)

except Exception as e:
    print("An error occurred:", e)