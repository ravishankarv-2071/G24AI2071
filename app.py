from flask import Flask, request, render_template_string
import joblib
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("savedmodel.pth")

# Simple HTML template
HTML = '''
<!doctype html>
<title>Olivetti Face Prediction</title>
<h1>Upload a face image index (0-399)</h1>
<form method=post>
  Index: <input type=number name=index min=0 max=399>
  <input type=submit value=Predict>
</form>
{% if prediction is not none %}
<h2>Predicted Class: {{ prediction }}</h2>
{% endif %}
'''

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        idx = int(request.form["index"])
        data = fetch_olivetti_faces()
        X = data.data[idx].reshape(1, -1)
        prediction = model.predict(X)[0]
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
