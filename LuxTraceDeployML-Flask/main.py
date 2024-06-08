from flask import Flask, request, jsonify
from model import preprocess_anomaly_data, predict_anomaly, preprocess_forecasting_data, predict_forecasting
import json
import db

app = Flask(__name__)

@app.route('/get_data', methods=['GET'])
def get_data():
    table_name = request.args.get('table', 'your_table')
    return db.get(table_name)

@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    prediction = predict_anomaly(data)
    record = {
        'input_data': json.dumps(data),
        'prediction_result': json.dumps(prediction)
    }
    db.create('your_table', record)
    return jsonify({"anomalies": prediction})

@app.route('/predict_forecasting', methods=['POST'])
def predict_forecasting_endpoint():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    prediction = predict_forecasting(data)
    record = {
        'input_data': json.dumps(data),
        'prediction_result': json.dumps(prediction)
    }
    db.create('your_table', record)
    return jsonify({"forecasting": prediction})

if __name__ == '__main__':
    app.run(debug=True)
