
import joblib
import pandas as pd

def predict_semen_quality(extracted_data):
    """
    Takes extracted semen analysis data and returns the probability of a certain outcome.

    Parameters:
        extracted_data (dict): JSON data containing semen analysis results.

    Returns:
        float: Prediction probability from the model.
    """
    try:
        # Reload model every time to ensure it is fresh
        model = joblib.load('model.pkl')

        # Extract relevant features for prediction
        volume = extracted_data['semen_analysis']['volume']['value']
        concentration = extracted_data['semen_analysis']['concentration']['value']
        motility = extracted_data['semen_analysis']['motility']['value']

        # Create DataFrame for model input
        input_data = pd.DataFrame({
            'Volume': [volume],
            'Concentration': [concentration],
            'Motility': [motility]
        })

        # Ensure the input data has the correct format
        print(f"Input data: {input_data}")

        # Make prediction
        probability = model.predict_proba(input_data)[:, 1][0]

        return probability

    except KeyError as e:
        return f"Missing key in input data: {e}"
    except Exception as e:
        return f"Error processing data: {e}"
