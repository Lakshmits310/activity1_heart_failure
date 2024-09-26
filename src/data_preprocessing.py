# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the heart failure dataset."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """Preprocess the dataset: clean, scale, and split."""
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test