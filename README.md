# Predictive Model

Predict the winner of the English Premier League (EPL) using historical match data and machine learning techniques. This project leverages a Kaggle dataset to train a Random Forest Classifier, which simulates a season to forecast the most likely champion based on past performances.

# Table of Contents

Overview

Features

Getting Started

Prerequisites

Installation

Usage

Running the Prediction Script

Handling Encoding Issues

Dependencies

License

Acknowledgements

# Overview

The EPL Winner Prediction project aims to forecast the champion of the English Premier League by analyzing historical match data. Utilizing machine learning algorithms, specifically the Random Forest Classifier, the model is trained on past match results to predict future outcomes. The program simulates an entire season, assigning points based on predicted match results to determine the most probable winner.

# Features
Data Preprocessing: Handles missing values and encodes categorical variables for model compatibility.
Feature Engineering: Creates additional features such as goal difference to enhance prediction accuracy.
Model Training: Utilizes a Random Forest Classifier for robust and accurate predictions.
Season Simulation: Simulates an entire EPL season based on predicted match outcomes to identify the likely winner.
Encoding Detection: Automatically detects and handles various file encodings to prevent data reading errors.
Getting Started
Follow these instructions to set up and run the EPL Winner Prediction program on your local machine.

# Prerequisites
Ensure you have the following installed on your system:

Python 3.6 or higher: Download Python  
pip: Python package installer (comes with Python)
Git (optional): For cloning the repository

# Installation
Clone the Repository (Optional):

bash

Copy code

git clone https://github.com/yourusername/predictive_model.git

cd predictive_model

Download the Dataset:

Obtain the EPL historical match data from Kaggle:
English Premier League Data
European Soccer Database
Download and place the CSV file (e.g., results.csv) in the project directory.
Create a Virtual Environment (Recommended):

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, you can install the necessary packages manually:

bash
Copy code
pip install pandas numpy scikit-learn chardet

# Usage
Running the Prediction Script
Ensure Dataset is in Place:

Place your results.csv file in the project directory.
Verify that the CSV file contains the following columns:
vbnet
Copy code
Season, DateTime, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR
Execute the Script:

bash
Copy code
python EPL_Prediction.py
Output:

The script will perform the following steps:

Detect and handle the CSV file's encoding.
Load and preprocess the data by selecting relevant columns, handling missing values, and encoding team names.
Engineer additional features.
Train a Random Forest model.
Evaluate the model's performance.
Simulate the season to predict the EPL winner.
Sample Output:

csharp
Copy code
Detected encoding: ISO-8859-1
Data loaded successfully with detected encoding.
Selected columns: ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
Checking for missing values...
HomeTeam    0
AwayTeam    0
FTHG        0
FTAG        0
FTR         0
dtype: int64
Data shape after dropping missing values: (3800, 5)
Model training completed.
Accuracy on test set: 0.60
Classification Report:
              precision    recall  f1-score   support

            -1       0.58      0.62      0.60        50
             0       0.62      0.59      0.60        30
             1       0.60      0.60      0.60        20

     accuracy                           0.60        100
    macro avg       0.60      0.60      0.60        100
 weighted avg       0.60      0.60      0.60        100

 The predicted Premier League winner is: Manchester United
Note: Actual results will vary based on the dataset and model performance.

# Handling Encoding Issues
When dealing with CSV files, especially those sourced from different platforms or regions, encoding issues like UnicodeDecodeError can arise. This project includes robust encoding detection and handling mechanisms to ensure smooth data loading.

Why Encoding Errors Occur
ASCII Encoding Limitation:
ASCII can only handle characters with byte values from 0 to 127. Characters beyond this range (e.g., non-breaking spaces represented by 0xa0) cause decoding errors.

Mixed or Incorrect Encodings:
The CSV file might be saved in an encoding different from what Python expects, leading to mismatches.

How the Script Handles Encoding
Automatic Detection with chardet:

The script uses the chardet library to detect the encoding of the CSV file automatically.

python
Copy code
import chardet

def detect_encoding(file_path, num_bytes=100000):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(num_bytes))
    return result['encoding']
Fallback Mechanism:

If the detected encoding fails to decode the file, the script attempts to read the file using 'latin1' encoding as a fallback.

python
Copy code
try:
    df = pd.read_csv(file_path, encoding=encoding)
    print("Data loaded successfully with detected encoding.")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError encountered:", e)
    print("Attempting to read with 'latin1' encoding.")
    df = pd.read_csv(file_path, encoding='latin1')
    print("Data loaded successfully with 'latin1' encoding.")
Selecting Relevant Columns:

After successfully loading the data, the script selects only the necessary columns to streamline processing.

python
Copy code
selected_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
df = df[selected_columns]
print(f"Selected columns: {selected_columns}")
Replacing or Ignoring Problematic Characters (Optional):

If further encoding issues persist, you can modify the pd.read_csv function to replace or ignore problematic characters.

python
Copy code
df = pd.read_csv('results.csv', encoding='utf-8', errors='replace')

df = pd.read_csv('results.csv', encoding='utf-8', errors='ignore')
Caution: Using errors='replace' or errors='ignore' can lead to data loss or corruption.

# Dependencies
The project relies on several Python libraries for data manipulation, machine learning, and encoding detection. Ensure all dependencies are installed using the provided requirements.txt.

Python Libraries
pandas: Data manipulation and analysis.
numpy: Numerical operations.
scikit-learn: Machine learning algorithms and tools.
chardet: Encoding detection.
Installing Dependencies
If you have a requirements.txt file, install dependencies using:

bash
Copy code
pip install -r requirements.txt
Alternatively, install them manually:

bash
Copy code
pip install pandas numpy scikit-learn chardet
requirements.txt Example
Create a requirements.txt file with the following content:

shell
Copy code
pandas>=1.1.5
numpy>=1.19.5
scikit-learn>=0.24.2
chardet>=4.0.0

# License
This project is licensed under the MIT License.

# Acknowledgements
Kaggle: For providing comprehensive datasets.
scikit-learn: For their powerful machine learning library.
chardet: For encoding detection.
