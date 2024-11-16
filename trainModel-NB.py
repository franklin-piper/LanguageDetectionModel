import csv
import json
import os
import threading
import joblib
import pandas as pd
import lzma
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
import re
from tqdm import tqdm  # Import tqdm for progress bars

# Define paths for model and vectorizer
model_path = 'language_model.pkl'
vectorizer_path = 'vectorizer.pkl'

# JSON for defining languages to use
languages_file = 'language_config.json'

# Create data, labels and compressed_files as global variable to allow space saving measures
data = []
labels = []
compressed_files = []

# Ensure the directory for the model path exists
model_dir = os.path.dirname(model_path)
if model_dir:
    os.makedirs(model_dir, exist_ok=True)

# Initialize the model and vectorizer
model = MultinomialNB()
vectorizer = CountVectorizer(ngram_range=(1, 3))  # Vectorizer will be fitted later

# Timing Data Lists
fitting_timing_data = []
transform_timing_data = []

# Determine number of languages in JSON file
with open(languages_file, 'r', encoding='utf-8') as file:
    json_data = json.load(file) # Parse JSON into dictionary

# Get number of entries in the "languages" array
num_languages = len(json_data.get("languages", [])) # Defaults to empty list if 'languages' keys is empty

# Create a barrier that will block until all threads reach it
barrier = threading.Barrier(num_languages)  

# Function to download a language file
def download_language_file(url, output_file):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(output_file, 'wb') as file, tqdm(
        desc=f"Downloading {output_file}",
        total=total_size,
        unit='B',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(1024):
            file.write(data)
            progress_bar.update(len(data))
    print(f"\nDownloaded {output_file}")

    # Wait for all threads to finish their work
    barrier.wait()

# Function to extract .xz files
def extract_xz_file(input_file, output_file, index, log_file="extraction_timing_data.csv"):
    print(f"\nExtracting {input_file}")
    # Get total file size in bytes
    total_size = os.path.getsize(input_file)
    chunk_size = 1024 * 1024  # 1 MB

    # Record start time
    start_time = time.time()

    with lzma.open(input_file, 'rb') as file, open(output_file, 'wb') as out_file, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=f"Extracting {input_file}",
        dynamic_ncols=True  # Adjusts width dynamically for more space
    ) as progress_bar:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            out_file.write(chunk)
            progress_bar.update(len(chunk))

    # Record end time and calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time  # Total time in seconds

    # Log data for training
    with open(log_file, mode='a', newline='') as log:
        writer = csv.writer(log)
        # Write header if file is empty
        if log.tell() == 0:
            writer.writerow(["File Size (bytes)", "Chunk Size (bytes)", "Time Taken (seconds)"])
        writer.writerow([total_size, chunk_size, time_taken])
    
    print("Extraction Completed")

    # Load File
    load_limited_bytes_from_files(output_file, labels[index])

    # Remove the no longer needed files
    os.remove(output_file)

    # Wait for all threads to finish their work
    barrier.wait()

# Function to load languages configuration
def load_languages_from_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return [(entry["label"], entry["url"]) for entry in data["languages"]]

# Step 1: Download and extract all files, then load data into a single DataFrame
def load_data():
    all_data = []
    languages = load_languages_from_json(languages_file)
    threads= []
    extracted_files = []

    # Download Files
    for index, (label, url) in enumerate(languages):

        compressed_file = f"{label.lower()}_texts.xz"
        extracted_file = f"{label.lower()}_texts"

        compressed_files.append(compressed_file)
        extracted_files.append(extracted_file)
        labels.append(label)

        # Download files
        thread = threading.Thread(target=download_language_file, args=(url, compressed_files[index]))
        threads.append(thread)
        thread.start()

    # Wait for all download threads to complete
    for thread in threads:
        thread.join() 

    # Delete old threads
    threads.clear()
    
    # Extract files
    for index, compressed_file in enumerate(compressed_files):
        #Extract files
        thread = threading.Thread(target=extract_xz_file, args=(compressed_file, extracted_files[index], index))
        threads.append(thread)
        thread.start()

    # Wait for all extraction threads to complete
    for thread in threads:
        thread.join() 

    # Delete old threads
    threads.clear()
 
    # Add data to DataFrame
    df = pd.DataFrame(data, columns=['text', 'language'])

    # Append to list
    all_data.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    full_data = pd.concat(all_data, ignore_index=True)
    return full_data  # Return the combined data

def format_time(seconds):
    # Format time in seconds to hours:minutes:seconds.
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}:{minutes:02}:{seconds:02}"

def adjusted_time_estimate(estimated_time, model_file):
    # Load the trained model
    model = joblib.load(model_file)
    
    # Predict the adjusted time
    adjusted_time = model.predict([[estimated_time]])[0]
    return adjusted_time

def normalize_text(text):
    # Keep only letters from any language and some common symbols (optional)
    normalized_text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)  # \p{L} matches any letter
    normalized_text = re.sub(r'-?\d*\.?\d+', '', normalized_text) # Remove numbers
    normalized_text = re.sub(r'\s+', ' ', normalized_text)  # Replace multiple spaces with a single space
    return normalized_text.strip()  # Remove leading/trailing spaces

def load_limited_bytes_from_files(file_path, label, max_bytes=1_000_000):  # Limit to 1MB per file
    with open(file_path, 'r', encoding='utf-8') as file:  # Open as text file
        print(f"Processing {label}...")
        content = file.read(max_bytes)  # Read only the specified number of bytes
        normalized_text = normalize_text(content)  # Normalize the text
        if normalized_text:  # Add only if not empty
            data.append((normalized_text, label))
    print("Processing Complete")

# Function to train the model on the full dataset with loading animation
def train_model_on_full_data(data):
    print("Building sample...")
    # Sample for timing estimation
    sample_size = min(10000, len(data))  # Use a sample size that is reasonable
    sample_data = data.sample(n=sample_size, random_state=42)

    # Measure time for fitting sample
    print("Fitting sample...")
    start_time = time.time()
    vectorizer.fit(sample_data['text'])
    sample_time = time.time() - start_time

    # Estimate time for full dataset
    estimated_time = (sample_time / sample_size) * len(data)

    # Adjust the time estimate
    adjusted_time = adjusted_time_estimate(estimated_time, 'fitting_time_adjustment_model.pkl')

    # Format time estimate for output
    formatted_adjusted_time = format_time(adjusted_time)
    print(f"Estimated time to fit vocabulary on the full dataset: {formatted_adjusted_time} (HH:MM:SS).")

    # Simulate the loading process with a progress bar
    for _ in tqdm(range(100), desc="Fitting vocabulary on dataset...", leave=True):
        time.sleep(adjusted_time / 100)  # Simulate loading time - add 5 minutes, fixes discrepancy

    # Fit the vectorizer on the entire dataset
    start_time = time.time()  # Start time for fitting full data
    vectorizer.fit(data['text'])
    joblib.dump(vectorizer, vectorizer_path)
    actual_time = time.time() - start_time
    formatted_actual_time = format_time(actual_time)
    print(f"Vocabulary initialized and saved in {formatted_actual_time}.")

    # Store (estimated, actual) time in fitting timing data list
    fitting_timing_data.append((adjusted_time,actual_time))

    # Save timing data to a CSV for future analysis
    pd.DataFrame(fitting_timing_data, columns=['Estimated', 'Actual']).to_csv('fitting_timing_data.csv', index=False)

    # Sample for timing estimation for transforming
    sample_data = data.sample(n=sample_size, random_state=42)  # Reuse the sample data for consistency
    start_time = time.time()
    vectorizer.transform(sample_data['text'])
    sample_transform_time = time.time() - start_time

    # Estimate time for full transformation
    estimated_time_transform = (sample_transform_time / sample_size) * len(data)

    # Adjust estimated time
    adjusted_time_transform = adjusted_time_estimate(estimated_time_transform, 'transform_time_adjustment_model.pkl')

    formatted_adjusted_time_transform = format_time(adjusted_time_transform)
    print(f"Estimated time to transform the full dataset: {formatted_adjusted_time_transform} (HH:MM:SS).")

    # Simulate the loading process with a progress bar
    for _ in tqdm(range(100), desc="Transforming dataset for training...", leave=True):
        time.sleep((estimated_time_transform + 300) / 100)  # Simulate loading time - add 5 minutes, fixes discrepancy

    # Transform data
    start_time = time.time()  # Start time for transforming full data
    X = vectorizer.transform(data['text'])
    y = data['language']
    actual_time_transform = time.time() - start_time
    formatted_actual_time_transform = format_time(actual_time_transform)
    print(f"Data transformed in {formatted_actual_time_transform} (HH:MM:SS).")

    # Store (estimated, actual) time in transform timing data list
    transform_timing_data.append((adjusted_time_transform,actual_time_transform))

    # Save timing data to a CSV for future analysis
    pd.DataFrame(transform_timing_data, columns=['Estimated', 'Actual']).to_csv('transform_timing_data.csv', index=False)

    # Display progress for model training
    batch_size = 10000  # Adjust batch size as needed
    num_batches = (len(y) + batch_size - 1) // batch_size  # Calculate total number of batches

    try:
        print("Training the model...")
        
        # Get unique language labels
        all_languages = sorted(full_data['language'].unique())

        # Replace with the full list of languages
        for i in tqdm(range(num_batches), desc="Training Progress"):
            start = i * batch_size
            end = start + batch_size
            model.partial_fit(X[start:end], y[start:end], classes=all_languages)

        print("Model training completed and saved.")
    finally:
        print("Model training complete!")  # Replace "Training the model..." with a complete message

    # Save the trained model with xz compression
    joblib.dump(model, model_path, compress=('xz', 3))

# Main execution flow
print("Starting data loading from text files...")
full_data = load_data()         # Load and combine all data
print("Data loaded. Starting model training...")
train_model_on_full_data(full_data)      # Train model with consistent features
print("Training complete and model saved.")
# Delete compressed files
for comp_file in compressed_files:
    os.remove(comp_file)