from flask import Flask, request, jsonify
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)


# Load and train the model
def train_model():
    data = load_iris()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')


# Train the model when the application starts
train_model()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        features = pd.DataFrame(data)
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('model.joblib')
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
