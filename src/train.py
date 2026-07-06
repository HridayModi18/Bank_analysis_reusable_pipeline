import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def train_and_evaluate(X_train, y_train, X_val, y_val, feature_names):
    """Train multiple models, log metrics to MLflow, and return the best model."""
    
    candidate_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42)
    }
    
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    evaluation_logs = {}

    mlflow.set_experiment("Bank_Customer_Churn_Prediction")

    for model_name, model in candidate_models.items():
        print(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)
            mlflow.log_params(model.get_params() if hasattr(model, 'get_params') else {})
            mlflow.log_metric("validation_f1_score", f1)
            mlflow.log_metric("validation_precision", precision)
            mlflow.log_metric("validation_recall", recall)
            mlflow.log_metric("validation_accuracy", accuracy)
            



            if "XGBoost" in model_name:
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
            
            print(f"{model_name} Validation F1-Score: {f1:.4f}")
            evaluation_logs[model_name] = f1
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_model_obj = model

    print(f"\nWinning model selected: {best_model_name} with F1-Score of {best_f1:.4f}")
    return best_model_obj, best_model_name

def get_original_value(feature_name, customer_id, active_df, scaled_val):
    """Retrieve original unscaled values or binary indicator values for one-hot features."""
    try:
        if feature_name in active_df.columns:
            val = active_df.loc[active_df['customer_id'] == customer_id, feature_name].values[0]
            if pd.isna(val):
                return None
            return float(val)

        prefixes = ['gender_', 'cidade_', 'country_', 'age_band_']
        for prefix in prefixes:
            if feature_name.startswith(prefix):
                base_column = prefix[:-1]  # strip the trailing underscore
                val = active_df.loc[active_df['customer_id'] == customer_id, base_column].values[0]
                category_suffix = feature_name[len(prefix):].lower()
                if str(val).strip().lower() == category_suffix:
                    return 1.0
                else:
                    return 0.0
    except Exception:
        pass
    return round(float(scaled_val), 4)

def generate_shap_insights(best_model, X_active_processed, active_df, active_customer_ids, feature_names):
    print("Generating SHAP explainability insights...")
    try:
        if isinstance(best_model, LogisticRegression):
            explainer = shap.LinearExplainer(best_model, X_active_processed)
        else:
            explainer = shap.TreeExplainer(best_model)
        
        shap_values = explainer.shap_values(X_active_processed)
    except Exception as error:
        print(f"SHAP Explainer fall back to default Explainer due to: {error}")
        explainer = shap.Explainer(best_model, X_active_processed)
        shap_values = explainer(X_active_processed).values

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    


    pred_probabilities = best_model.predict_proba(X_active_processed)[:, 1]
    
    records = []
    for index, customer_id in enumerate(active_customer_ids):
        customer_shap = shap_values[index]
        customer_features_val = X_active_processed[index]
        
        sorted_indices = np.argsort(customer_shap)[::-1]
        
        top_1_idx = sorted_indices[-1]
        top_2_idx = sorted_indices[0]
        
        top_1_feat_name = feature_names[top_1_idx]
        top_2_feat_name = feature_names[top_2_idx]
        
        top_1_orig_val = get_original_value(top_1_feat_name, customer_id, active_df, customer_features_val[top_1_idx])
        top_2_orig_val = get_original_value(top_2_feat_name, customer_id, active_df, customer_features_val[top_2_idx])
        
        records.append({
            "customer_id": int(customer_id),
            "churn_probability": round(float(pred_probabilities[index]), 4),
            "top_1_name": top_1_feat_name,
            "top_1_value": top_1_orig_val,
            "top_1_impact": round(float(customer_shap[top_1_idx]), 4),
            "top_2_name": top_2_feat_name,
            "top_2_value": top_2_orig_val,
            "top_2_impact": round(float(customer_shap[top_2_idx]), 4)
        })
        
    insights_df = pd.DataFrame(records)
    insights_df.to_csv("dynamic_insights.csv", index=False)
    print("Exported top drivers to dynamic_insights.csv")
    global_importances = np.mean(np.abs(shap_values), axis=0)
    raw_importance_df = pd.DataFrame({
        "feature_name": feature_names,
        "global_importance": np.round(global_importances, 6)
    }).sort_values(by="global_importance", ascending=False)

    raw_importance_df.to_csv("raw_global_feature_importance.csv", index=False)
    print(f"Exported raw global feature importances of {len(feature_names)} features to raw_global_feature_importance.csv")

    ohe_prefixes = ["country_", "cidade_", "gender_", "age_band_"]

    grouped_importance: dict[str, float] = {}
    for feat, imp in zip(feature_names, global_importances):
        original = feat
        for prefix in ohe_prefixes:
            if feat.startswith(prefix):
                original = prefix[:-1]  # strip trailing underscore → e.g. "country"
                break
        grouped_importance[original] = grouped_importance.get(original, 0.0) + imp

    global_importance_df = pd.DataFrame(
        list(grouped_importance.items()),
        columns=["feature_name", "global_importance"]
    )
    global_importance_df["global_importance"] = np.round(
        global_importance_df["global_importance"], 6
    )
    global_importance_df = global_importance_df.sort_values(
        by="global_importance", ascending=False
    ).reset_index(drop=True)

    global_importance_df.to_csv("global_feature_importance.csv", index=False)
    print(
        f"Exported aggregated global feature importances "
        f"({len(global_importance_df)} original features) to global_feature_importance.csv"
    )
    return insights_df
