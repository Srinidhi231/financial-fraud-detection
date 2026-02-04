# Fraud Detection System

This project implements a machine learning-based fraud detection system that can identify fraudulent transactions using a trained model and a user-friendly web interface.

## Project Overview

The system uses a Random Forest classifier to predict the probability of fraud in financial transactions. It includes:
- A trained machine learning model for fraud detection
- A Streamlit web application for easy interaction
- Scripts to generate sample data for testing
- Preprocessing tools for data normalization

## File Structure

```
├── app.py                    # Streamlit web application for fraud detection
├── sam.py                    # Model training and preprocessing script
├── make_fraud_sample.py      # Script to create fraud sample datasets
├── make_test_sample.py       # Script to create test sample datasets
├── fraud_model.joblib        # Trained Random Forest model
├── categorical_mappings.json # Categorical feature mappings
├── feature_columns.json      # List of feature columns used in training
├── README.md                 # This file
├── sample_submission.csv     # Sample Kaggle submission file
├── submission.csv            # Generated submission file
├── train_transaction.csv     # Training transaction data
├── train_identity.csv        # Training identity data
├── test_transaction.csv      # Test transaction data
├── test_identity.csv         # Test identity data
├── my_fraud_sample.csv       # Sample fraud-only dataset
├── my_test_input.csv         # Sample test dataset
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
```bash
pip install streamlit pandas joblib scikit-learn
```

## Usage

### 1. Web Application

Run the Streamlit web application to detect fraud through a user-friendly interface:

```bash
streamlit run app.py
```

The application will:
- Allow you to upload a CSV file with transaction data
- Process the data using the trained model
- Display fraud probabilities and predictions
- Enable downloading of results as a CSV file

### 2. Command Line Prediction

Use the trained model directly from the command line:

```bash
python sam.py --predict_csv your_data.csv --output predictions.csv
```

### 3. Generate Sample Data

Create sample datasets for testing:

```bash
# Generate a fraud sample dataset
python make_fraud_sample.py --nrows 500

# Generate a test sample dataset
python make_test_sample.py --nrows 1000
```

## Model Training

To retrain the model with your own data:

```bash
python sam.py
```

This will:
- Load and preprocess the training data
- Train a Random Forest classifier
- Evaluate the model on a validation set
- Save the trained model and preprocessing artifacts

## Data Format

The system expects CSV files with the following characteristics:
- Must contain transaction data merged with identity data
- Column names with hyphens (e.g., "id-01") will be normalized to underscores (e.g., "id_01")
- Must include all features used during training
- Missing values will be filled with -999

## Requirements

- Python 3.7+
- streamlit
- pandas
- scikit-learn
- joblib

## License

This project is for educational purposes. Please check the licensing terms of the original dataset before using this code for commercial purposes.