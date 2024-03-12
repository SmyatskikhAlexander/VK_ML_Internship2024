from flask import Flask, jsonify, request
import pandas as pd
import pickle
from catboost import CatBoostClassifier, Pool
import os
import json

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():  
    file = request.files['123.csv']
    filepath = os.path.join('/app/', file.filename)
    file.save(filepath)
    df = pd.read_csv(file.filename)
    predictions = model.predict(df)
    y_dict = [{'prediction': str(pred)} for pred in predictions]
    return jsonify({'result':y_dict})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)