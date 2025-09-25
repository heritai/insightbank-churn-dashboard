"""
Data preparation utilities for InsightBank dashboard
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_customer_data(file_path='sample_data/customers.csv'):
    """Load customer data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} customer records")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find data file at {file_path}")
        return None

def clean_data(df):
    """Clean and validate customer data"""
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check for missing values
    missing_values = df_clean.isnull().sum()
    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        # Fill missing values with appropriate defaults
        df_clean = df_clean.fillna({
            'Age': df_clean['Age'].median(),
            'Income': df_clean['Income'].median(),
            'Tenure': df_clean['Tenure'].median(),
            'Number_of_Products': df_clean['Number_of_Products'].median(),
            'Activity_Score': df_clean['Activity_Score'].median()
        })
    
    # Ensure data types are correct
    df_clean['Age'] = df_clean['Age'].astype(int)
    df_clean['Income'] = df_clean['Income'].astype(int)
    df_clean['Number_of_Products'] = df_clean['Number_of_Products'].astype(int)
    df_clean['Activity_Score'] = df_clean['Activity_Score'].astype(float)
    df_clean['Tenure'] = df_clean['Tenure'].astype(float)
    
    # Validate data ranges
    df_clean = df_clean[
        (df_clean['Age'] >= 18) & (df_clean['Age'] <= 100) &
        (df_clean['Income'] >= 0) & (df_clean['Income'] <= 500000) &
        (df_clean['Tenure'] >= 0) & (df_clean['Tenure'] <= 50) &
        (df_clean['Number_of_Products'] >= 1) & (df_clean['Number_of_Products'] <= 20) &
        (df_clean['Activity_Score'] >= 0) & (df_clean['Activity_Score'] <= 100)
    ]
    
    print(f"Data cleaned. {len(df_clean)} records remaining after validation.")
    return df_clean

def prepare_features(df):
    """Prepare features for machine learning models"""
    df_features = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df_features['Gender_Encoded'] = le_gender.fit_transform(df_features['Gender'])
    
    # Create binary churn variable for modeling
    df_features['Churn_Binary'] = (df_features['Churn'] == 'Yes').astype(int)
    
    # Create additional features
    df_features['Income_per_Product'] = df_features['Income'] / df_features['Number_of_Products']
    df_features['Activity_per_Tenure'] = df_features['Activity_Score'] / (df_features['Tenure'] + 1)
    df_features['Age_Group'] = pd.cut(df_features['Age'], 
                                    bins=[0, 30, 50, 70, 100], 
                                    labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # Encode age group
    le_age = LabelEncoder()
    df_features['Age_Group_Encoded'] = le_age.fit_transform(df_features['Age_Group'])
    
    return df_features, le_gender, le_age

def get_feature_columns():
    """Get list of feature columns for modeling"""
    return [
        'Age', 'Income', 'Tenure', 'Number_of_Products', 
        'Activity_Score', 'Gender_Encoded', 'Income_per_Product',
        'Activity_per_Tenure', 'Age_Group_Encoded'
    ]

def prepare_training_data(df, test_size=0.2, random_state=42):
    """Prepare training and testing datasets"""
    feature_cols = get_feature_columns()
    
    X = df[feature_cols]
    y = df['Churn_Binary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_cols
    }

def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_customers = len(df)
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    avg_tenure = df['Tenure'].mean()
    avg_income = df['Income'].mean()
    avg_activity = df['Activity_Score'].mean()
    avg_products = df['Number_of_Products'].mean()
    
    return {
        'total_customers': total_customers,
        'churn_rate': churn_rate,
        'avg_tenure': avg_tenure,
        'avg_income': avg_income,
        'avg_activity': avg_activity,
        'avg_products': avg_products
    }

def get_customer_summary(df, customer_id):
    """Get summary information for a specific customer"""
    customer = df[df['CustomerID'] == customer_id]
    if customer.empty:
        return None
    
    customer = customer.iloc[0]
    return {
        'CustomerID': customer['CustomerID'],
        'Age': customer['Age'],
        'Gender': customer['Gender'],
        'Income': customer['Income'],
        'Tenure': customer['Tenure'],
        'Number_of_Products': customer['Number_of_Products'],
        'Activity_Score': customer['Activity_Score'],
        'Churn': customer['Churn']
    }
