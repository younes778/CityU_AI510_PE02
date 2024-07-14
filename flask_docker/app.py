from flask import Flask, render_template
import os

app = Flask(__name__)
import joblib
poly1d_clf = joblib.load('poly1d_clf.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.get('/predict/<int:input_data>')
def converter_with_type(input_data):
    pred_data = [[input_data]]
    predict_result = poly1d_clf(pred_data)
    return ("Prediction result: " + str(predict_result))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 3456))
    app.run(debug=True, host='0.0.0.0', port=port)