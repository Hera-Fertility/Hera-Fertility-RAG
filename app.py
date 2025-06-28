
from flask import Flask, request, jsonify
import os
import pandas as pd
from pdf_processor import extract_structured_data  # PDF extraction
from prediction import predict_semen_quality  # Prediction function
from explanation import generate_shap_explanation  # Explanation generation
from run_rag_from_json import generate_rag_explanation  #
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """Handles PDF upload and extraction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        # Extract structured data from PDF
        data = extract_structured_data(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)  # Cleanup uploaded file

@app.route('/predict', methods=['POST'])
def predict():
    """Takes extracted semen analysis data and returns prediction probability."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Invalid or missing JSON data"}), 400

    result = predict_semen_quality(data)
    return jsonify(result)


# def explain():
#     """Returns a Gemini-generated explanation of the model's prediction."""
#     json_data = request.get_json()
#     if not json_data:
#         return jsonify({"error": "Missing JSON input"}), 400

#     try:
#         explanation_text = generate_explanation(json_data)
#         return jsonify({"explanation": explanation_text})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
@app.route('/explain', methods=['POST'])
def explain():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Missing JSON input"}), 400

    try:
        shap_result = generate_shap_explanation(json_data)
        return jsonify(shap_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     json_data = request.get_json()
#     if not json_data:
#         return jsonify({"error": "Missing JSON input"}), 400

#     try:
#         shap_json = generate_shap_explanation(json_data)
#         rag_result = generate_rag_explanation(shap_json)
#         return jsonify({"recommendation": rag_result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "Missing JSON input"}), 400

    try:
        shap_json = generate_shap_explanation(json_data)
        rag_result = generate_rag_explanation(shap_json)

        return jsonify({
            "recommendation": rag_result,
            "fertility_score": shap_json.get("fertility_score"),
            "features": shap_json.get("features")  # include SHAP for frontend plot
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
