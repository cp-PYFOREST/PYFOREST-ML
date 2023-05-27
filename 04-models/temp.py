import joblib

# Create a dictionary
data = {'a': 1, 'b': 2, 'c': 3}

# Use joblib to save the dictionary to a file
try:
    joblib.dump(data, 'data.pkl')
    print("Data saved successfully.")
except Exception as e:
    print(f"An error occurred while saving data: {e}")

# Use joblib to load the dictionary from the file
try:
    loaded_data = joblib.load('data.pkl')
    print("Data loaded successfully.")
    print("Loaded data:", loaded_data)
except Exception as e:
    print(f"An error occurred while loading data: {e}")
