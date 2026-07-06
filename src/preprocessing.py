import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def get_feature_columns():
    categorical_features = ['gender', 'cidade', 'country']
    numerical_features = [
        'credit_score' , 'age', 'tenure', 'balance', 'limite_credito', 
        'products_number', 'credit_card', 'active_member', 'estimated_salary',
        'credit_utilization', 'salary_missing_flag'
    ]
    return categorical_features, numerical_features

def build_preprocessing_pipeline():
    categorical_features, numerical_features = get_feature_columns()
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])



    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def prepare_training_data(df):
    train_df = df[df['churn'].notna()].copy()
    
    
    
    X = train_df.drop(columns=['customer_id', 'nome', 'email', 'numero_conta', 'data_nascimento', 'churn'])
    y = train_df['churn'].astype(int)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_encoded_names = list(cat_encoder.get_feature_names_out(get_feature_columns()[0]))
    final_feature_names = get_feature_columns()[1] + cat_encoded_names
    
    
    # smote was needed to learn for balacing th data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print(f"Original training shape: {X_train_processed.shape}, Resampled shape: {X_train_resampled.shape}")
    
    return {
        'X_train': X_train_resampled,
        'y_train': y_train_resampled,
        'X_val': X_val_processed,
        'y_val': y_val,
        'preprocessor': preprocessor,
        'feature_names': final_feature_names
    }

def prepare_prediction_data(df, preprocessor):
    active_df = df[df['churn'].isna()].copy()
    if active_df.empty:
        print("No records with NULL churn found. Falling back to active customers (churn = 0).")
        active_df = df[df['churn'] == 0].copy()
    if active_df.empty:
        print("No active customers to predict churn for.")
        return None, None   
    X_active = active_df.drop(columns=['customer_id', 'nome', 'email', 'numero_conta', 'data_nascimento', 'churn'])
    X_active_processed = preprocessor.transform(X_active)
    return active_df['customer_id'].values, X_active_processed
