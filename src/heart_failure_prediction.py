# heart_failure_prediction.py

from data_preprocessing import load_data, preprocess_data
from model_training import train_model, evaluate_model
from model_io import save_model, load_model

def main():
    # Load the data
    data = load_data('heart_failure_clinical_records_dataset.csv')
    
    # Preprocess data
    target_column = 'DEATH_EVENT'  # Target in the dataset
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the model
    save_model(model, 'heart_failure_model.pkl')
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Classification Report:\n{report}')
    
if __name__ == '__main__':
    main()