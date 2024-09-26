# Heart Failure Prediction

This project is focused on predicting heart failure based on a clinical records dataset using machine learning techniques. The model can be trained on patient records to predict the likelihood of heart failure.

## Project Structure

```bash
heart_failure_prediction/
│
├── data_preprocessing.py        # Data Preprocessing Module
├── model_training.py            # Model Training Module
├── model_io.py                  # Model Saving and Loading Module
├── heart_failure_prediction.py  # Main Script (orchestrates the process)
├── requirements.txt             # List of dependencies for the project
├── heart_failure_clinical_records_dataset.csv  # Dataset used for training
├── README.md                    # Project Documentation
└── models/                      # Directory to store trained models
    └── heart_failure_model.pkl  # Serialized model (created after running the script)