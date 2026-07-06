
import pandas as pd
import numpy as np

def generate_predictions(best_model, X_active_processed, active_customer_ids, original_full_df):
    print("Generating churn predictions for active customers...")
    
    #this predict binary outcomes and continuous probabilities , when probablity more than 0.5 then 1
    probabilities = best_model.predict_proba(X_active_processed)[:, 1]
    class_predictions = best_model.predict(X_active_processed)
    # we will extract the active users that is with churn==0
    active_customers_df = original_full_df[original_full_df['churn'].isna()].copy()
    if active_customers_df.empty:
        active_customers_df = original_full_df[original_full_df['churn'] == 0].copy()
    
    # probablity up to 4 decimal
    active_customers_df['churn_probability'] = np.round(probabilities, 4)
    active_customers_df['churn_predicted_class'] = class_predictions
    


    # for csv making 
    active_customers_df.to_csv("prediction_results.csv", index=False)
    print(f"Exported prediction results to prediction_results.csv for {len(active_customers_df)} customers.")
    
    return active_customers_df
