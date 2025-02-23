from flask import Flask, request, jsonify
import pandas as pd
import joblib
from controllers.preprocessing import creer_df, df_transformer, align_columns
from models.read_files import split_columns, get_logs
#from models.predictions import predict


app = Flask(__name__)

# Model laden
model = joblib.load("../models/model.pkl")
exp_col = pd.read_csv("../models/expected_columns.csv", header=None).squeeze("columns").tolist()

@app.route('/predict_file', methods=['POST'])
def predict_from_file():
    if 'logfile' not in request.files:
        return jsonify({"error": "Aucun fichier telecharge"}), 400

    file = request.files['logfile']
    logs = get_logs(file)
    logs = split_columns(logs)

    # Features extraheren
    df = creer_df(logs)
    df = df_transformer(df)
    df = align_columns(df, expected_columns=exp_col)

    # Voorspelling maken
    prediction = model.predict(df)

    # Resultaat terugsturen als JSON
    result = {"predicted_scores": prediction.tolist()}
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
