from flask import Flask, render_template, request
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text


# Load mô hình và label encoder
model = joblib.load(r"D:\Personal Project\text-topic-classifier\outputs\text_classifier.pkl")
label_encoder = joblib.load(r"D:\Personal Project\text-topic-classifier\outputs\label_encoder.pkl")

# Khởi tạo Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "")
        if input_text.strip():
            cleaned = clean_text(input_text)
            pred = model.predict([cleaned])
            prediction = label_encoder.inverse_transform(pred)[0]

    return render_template("index.html", prediction=prediction, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
