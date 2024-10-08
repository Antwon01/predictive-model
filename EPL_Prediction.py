import pandas as pd
import numpy as np
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def detect_encoding(file_path, num_bytes=100000):
    
    #Detect the encoding of a file using chardet.

    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(num_bytes))
    return result['encoding']

def load_data(file_path):
    
    #Load CSV data with robust encoding handling and select relevant columns.

    # Detect encoding
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    # List of encodings to try
    encodings_to_try = [encoding, 'latin1', 'cp1252', 'utf-8-sig', 'utf-8']

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"Data loaded successfully with encoding: {enc}")
            break
        except UnicodeDecodeError as e:
            print(f"Failed to decode with encoding {enc}: {e}")
    else:
        raise ValueError("Failed to decode the file with tried encodings.")

    # Select only the relevant columns
    selected_columns = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    # Verify all selected columns exist in the dataframe
    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing in the dataset: {missing_columns}")
    
    df = df[selected_columns]
    print(f"Selected columns: {selected_columns}")
    
    return df

def preprocess_data(df):
    
    #Preprocess the DataFrame by handling missing values and encoding categorical variables.

    # Handle missing values
    print("Checking for missing values...")
    missing = df.isnull().sum()
    print(missing)
    
    # For simplicity, drop rows with any missing values
    df = df.dropna()
    print(f"Data shape after dropping missing values: {df.shape}")

    # Encode categorical variables (e.g., team names)
    le = LabelEncoder()
    df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeam'] = le.fit_transform(df['AwayTeam'])
    
    return df, le

def feature_engineering(df):
    
    #Create additional features that may help in prediction.

    # Create a goal difference feature
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        df['GoalDifference'] = df['FTHG'] - df['FTAG']
    else:
        raise KeyError("Columns 'FTHG' and/or 'FTAG' not found in the dataset.")
    
    return df

def define_target(df):
    
    #Define the target variable for the model.

    # Define target: Home Win = 1, Draw = 0, Away Win = -1
    def get_result(row):
        if row['FTR'] == 'H':
            return 1
        elif row['FTR'] == 'D':
            return 0
        else:
            return -1
    
    df['Result'] = df.apply(get_result, axis=1)
    return df['Result']

def split_data(features, target):
    
    #Split the data into training and testing sets.

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    
    #Train the Random Forest Classifier.

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    
    #Evaluate the trained model.

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.2f}")
    #print("Classification Report:")
    #print(classification_report(y_test, y_pred))

def predict_winner(df, model, le):
    
    #Predict the Premier League winner by simulating the season.

    teams = np.unique(np.concatenate((df['HomeTeam'], df['AwayTeam'])))
    team_points = {team: 0 for team in teams}

    # Shuffle the DataFrame to simulate a new season
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for index, row in df_shuffled.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        # Prepare match features as a DataFrame with correct column names
        match_features = pd.DataFrame([{
            'HomeTeam': row['HomeTeam'],
            'AwayTeam': row['AwayTeam'],
            'GoalDifference': row['GoalDifference']
        }])

        # Predict match result
        result = model.predict(match_features)[0]

        # Assign points based on result
        if result == 1:
            team_points[home_team] += 3
        elif result == 0:
            team_points[home_team] += 1
            team_points[away_team] += 1
        else:
            team_points[away_team] += 3

    # Determine the winner
    winner_id = max(team_points, key=team_points.get)
    winner_name = le.inverse_transform([winner_id])[0]
    print(f"The predicted Premier League winner is: {winner_name}")

def main():
    # Path to CSV file
    file_path = 'results.csv'

    # Load the data with encoding handling and select relevant columns
    df = load_data(file_path)

    # Preprocess the data
    df, le = preprocess_data(df)

    # Feature engineering
    df = feature_engineering(df)

    # Define target variable
    target = define_target(df)

    # Select features for the model
    features = df[['HomeTeam', 'AwayTeam', 'GoalDifference']]  # Add more features as needed

    # Split the data
    X_train, X_test, y_train, y_test = split_data(features, target)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Predict the EPL winner
    predict_winner(df, model, le)

if __name__ == "__main__":
    main()