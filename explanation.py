import shap
import pickle
import numpy as np
import pandas as pd

# Load model and background
with open("models/logistic regression.pkl", "rb") as f:
    package = pickle.load(f)
    model = package["model"]
    feature_names = package["feature_names"]

with open("models/background_data.pkl", "rb") as f:
    background = pickle.load(f)

explainer = shap.LinearExplainer(model, background)

def get_impact_strength(abs_impact):
    if abs_impact > 0.15:
        return "Very Strong"
    elif abs_impact > 0.10:
        return "Strong"
    elif abs_impact > 0.05:
        return "Moderate"
    else:
        return "Mild"

def generate_shap_explanation(extracted_data):
    volume = extracted_data['semen_analysis']['volume']['value']
    concentration = extracted_data['semen_analysis']['concentration']['value']
    motility = extracted_data['semen_analysis']['motility']['value']

    input_data = pd.DataFrame([{
        feature_names[0]: volume,
        feature_names[1]: concentration,
        feature_names[2]: motility
    }])

    probability = model.predict_proba(input_data)[0][1]
    shap_values = explainer.shap_values(input_data)

    feature_impact = pd.DataFrame({
        "Feature": feature_names,
        "Value": input_data.iloc[0].values,
        "SHAP_Impact": shap_values[0]
    })
    feature_impact["Abs_Impact"] = np.abs(feature_impact["SHAP_Impact"])

    # Build SHAP response
    shap_json = {
        "fertility_score": round(probability * 100, 2),
        "features": {}
    }

    for i, row in feature_impact.iterrows():
        shap_json["features"][row["Feature"]] = {
            "value": float(row["Value"]),
            "impact": float(row["SHAP_Impact"]),
            "impact_strength": get_impact_strength(row["Abs_Impact"]),
            "direction": "positive" if row["SHAP_Impact"] >= 0 else "negative"
        }

    return shap_json