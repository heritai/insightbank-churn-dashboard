"""
Churn prediction model using Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import joblib
import os

class ChurnPredictor:
    """Churn prediction model wrapper"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X_train, y_train, X_test, y_test, scaler):
        """Train the churn prediction model"""
        self.scaler = scaler
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError("Model type must be 'random_forest' or 'logistic_regression'")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        # Predictions for evaluation
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        self.is_trained = True
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_churn_probability(self, customer_data):
        """Predict churn probability for a single customer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure customer_data is a DataFrame with correct columns
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Select only the features used in training
        customer_features = customer_data[self.feature_names]
        
        # Scale the features
        customer_scaled = self.scaler.transform(customer_features)
        
        # Predict probability
        churn_probability = self.model.predict_proba(customer_scaled)[0, 1]
        
        return churn_probability
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if self.model_type == 'random_forest':
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            # For logistic regression, use absolute coefficients
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': np.abs(self.model.coef_[0])
            }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")

def create_customer_profile(age, gender, income, tenure, products, activity_score):
    """Create a customer profile for prediction"""
    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0
    
    # Create age group
    if age <= 30:
        age_group = 0  # Young
    elif age <= 50:
        age_group = 1  # Middle
    elif age <= 70:
        age_group = 2  # Senior
    else:
        age_group = 3  # Elderly
    
    # Calculate derived features
    income_per_product = income / products if products > 0 else 0
    activity_per_tenure = activity_score / (tenure + 1) if tenure > 0 else activity_score
    
    profile = {
        'Age': age,
        'Income': income,
        'Tenure': tenure,
        'Number_of_Products': products,
        'Activity_Score': activity_score,
        'Gender_Encoded': gender_encoded,
        'Income_per_Product': income_per_product,
        'Activity_per_Tenure': activity_per_tenure,
        'Age_Group_Encoded': age_group
    }
    
    return profile

def interpret_churn_probability(probability):
    """Interpret churn probability with business context"""
    if probability < 0.3:
        risk_level = "Low Risk"
        interpretation = "Customer is likely to stay. Focus on retention and upselling."
        color = "green"
    elif probability < 0.6:
        risk_level = "Medium Risk"
        interpretation = "Customer shows some risk signs. Monitor closely and consider targeted retention efforts."
        color = "orange"
    else:
        risk_level = "High Risk"
        interpretation = "Customer is at high risk of churning. Immediate intervention recommended."
        color = "red"
    
    return {
        'risk_level': risk_level,
        'interpretation': interpretation,
        'color': color,
        'probability': probability
    }

def get_retention_recommendations(probability, customer_profile):
    """Generate specific retention recommendations based on churn probability and profile"""
    recommendations = []
    
    # High-level recommendations based on probability
    if probability > 0.6:
        recommendations.append("🚨 URGENT: Schedule immediate call with customer success team")
        recommendations.append("💰 Offer retention discount or special promotion")
        recommendations.append("📞 Conduct exit interview to understand pain points")
    elif probability > 0.4:
        recommendations.append("📧 Send personalized retention email")
        recommendations.append("🎯 Invite to customer success webinar")
        recommendations.append("💡 Offer product training session")
    else:
        recommendations.append("✅ Continue current engagement strategy")
        recommendations.append("📈 Focus on upselling opportunities")
    
    # Profile-specific recommendations
    if customer_profile['Activity_Score'] < 30:
        recommendations.append("📱 Increase engagement through mobile app notifications")
        recommendations.append("🎮 Introduce gamification elements")
    
    if customer_profile['Tenure'] < 1:
        recommendations.append("🎓 Enhance onboarding experience")
        recommendations.append("👥 Assign dedicated customer success manager")
    
    if customer_profile['Income'] < 30000:
        recommendations.append("💳 Offer budget-friendly product options")
        recommendations.append("📚 Provide financial education resources")
    
    if customer_profile['Number_of_Products'] < 2:
        recommendations.append("🔄 Cross-sell additional products")
        recommendations.append("📊 Show value of product bundling")
    
    return recommendations
