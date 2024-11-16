import os
import sys
import joblib
import pandas as pd
import time
from tqdm import tqdm

# Define paths for model and vectorizer in Google Drive
model_path = 'language_model.pkl'
vectorizer_path = 'vectorizer.pkl'

# Ensure the directory for the model path exists
model_dir = os.path.dirname(model_path)  # Get the directory from the model path
if model_dir:  # Check if model_dir is not an empty string
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
    print("Directory created")

print("Directory found")

# Estimate loading time based on file size
def estimate_loading_time(file_path, average_speed=5000000):  # Average speed in bytes per second
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        estimated_time = file_size / average_speed  # Time in seconds
        return estimated_time
    return 0

def format_time(seconds):
    #Format time in seconds to hours:minutes:seconds.
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return (f"{hours}:{minutes:02}:{seconds:02}")

# Estimate time to load model
estimated_model_time = estimate_loading_time(model_path)
estimated_vectorizer_time = estimate_loading_time(vectorizer_path)
total_estimated_time = estimated_model_time + estimated_vectorizer_time
formatted_total_estimated_time = format_time(total_estimated_time)

# Load or initialize the model and vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    print(f"Estimated loading time: {formatted_total_estimated_time}.")
    print("Loading Model...")

    # Simulate the loading process with a progress bar
    for _ in tqdm(range(100), desc="Loading model and vectorizer", leave=True):
        time.sleep((total_estimated_time + 300) / 100)  # Simulate loading time - add 5 minutes, fixes discrepancy

    start_time = time.time()  # Start timing
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    load_time = time.time() - start_time  # Calculate loading time
    formatted_load_time = format_time(load_time)
    print(f"Model Loaded in {formatted_load_time}")

else:
    sys.exit("ERROR: No model found. Please ensure you have provided the correct file path.")

# Function to predict the language and confidence of user input
def predict_language(model):
    while True:
        user_text = input("Enter text to detect its language (or type 'exit' to finish): ")
        if user_text.lower() == 'exit':
            break

        # Vectorize the input text
        user_text_vectorized = vectorizer.transform([user_text])

        # Make a prediction
        prediction = model.predict(user_text_vectorized)
        prediction_proba = model.predict_proba(user_text_vectorized)

        # Get the confidence score for the predicted language
        language = prediction[0]
        confidence = prediction_proba[0][model.classes_ == language][0]

        if confidence < 0.5:
            print("Predicted Language: Unknown Language")
            print(f"All probabilities: {prediction_proba}")  # Debugging line
            continue

        print(f"Predicted Language: {language}, Confidence: {confidence * 100:.2f}% ")
        print(f"All probabilities: {prediction_proba}")  # Debugging line

predict_language(model)