
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # input list

    features = np.array(data).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)



import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully!")

