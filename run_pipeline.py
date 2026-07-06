# run_pipeline.py
# Main orchestration script to run the entire Customer Churn pipeline with a single click.

import os
import sys
import pandas as pd

# Append src folder to path for package routing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import system pipeline modules
import db_connector
import preprocessing
import train
import predict
import generate_feature_impact_csvs

def main():
    print("-x-x-x-")
    print("Starting Bank Customer Churn Pipeline Execution")
    print("-x-x-x-")
    
    # Step 1: run the Database Quality Stored Procedure
    print("\n  Step 1: Executing database validation & cleaning procedure ---")
    db_connector.run_data_quality_procedure()
    
    # Step 2: extract data quality report and save it
    print("\n Step 2: Fetching data quality metrics ---")
    data_quality_df = db_connector.fetch_data_quality_metrics()
    data_quality_df.to_csv("data_quality_metrics.csv", index=False)
    print("Saved data health report to data_quality_metrics.csv")
    
    # Step 3: fetch clean analytics features view
    print("\n  Step 3: Fetching analytics features from DB view ---")
    features_raw_df = db_connector.fetch_analytics_features()
    
    # Step 4: Preprocess data and handle class balancing with SMOTE
    print("\n  Step 4: Preprocessing and balancing class distribution ---")
    training_data = preprocessing.prepare_training_data(features_raw_df)
    
    # Step 5: Run model training loop and select best model
    print("\n Step 5: Training model loop & selection ---")
    best_model, best_model_name = train.train_and_evaluate(
        X_train=training_data['X_train'],
        y_train=training_data['y_train'],
        X_val=training_data['X_val'],
        y_val=training_data['y_val'],
        feature_names=training_data['feature_names']
    )
    
    # Step 6: prepareing active customer prediction sets
    print("\n Step 6: Preparing prediction inputs for active customers ---")
    active_ids, X_active_processed = preprocessing.prepare_prediction_data(
        df=features_raw_df,
        preprocessor=training_data['preprocessor']
    )
    







    if active_ids is not None and len(active_ids) > 0:
        # churn predication
        print("\n Step 7: Running predictions and outputting prediction_results.csv ---")
        predict.generate_predictions(
            best_model=best_model,
            X_active_processed=X_active_processed,
            active_customer_ids=active_ids,
            original_full_df=features_raw_df
        )
        
        # making shap values
        print("\n Step 8: Running SHAP explanations and exporting top drivers ---")
        active_df_fallback = features_raw_df[features_raw_df['churn'].isna()]
        if active_df_fallback.empty:
            active_df_fallback = features_raw_df[features_raw_df['churn'] == 0]
            
        train.generate_shap_insights(
            best_model=best_model,
            X_active_processed=X_active_processed,
            active_df=active_df_fallback,
            active_customer_ids=active_ids,
            feature_names=training_data['feature_names']
        )

        # Step 9: ai made .py file using 
        print("\n Step 9: Generating feature impact CSVs for Power BI ---")
        generate_feature_impact_csvs.generate_feature_impact_csvs()
    else:
        print("\n---Step 6/7/8/9: was skipped bcoz No active customer records found")
        
    print("\n-x-x-x-")
    print("Pipeline Execution Completed Successfully!")
    print("-x-x-x-")

if __name__ == "__main__":
    main()
