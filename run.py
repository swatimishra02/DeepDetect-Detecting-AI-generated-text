import os
from flask import Flask, request, render_template
from app.model_utils import extract_features_from_text, predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        features = extract_features_from_text(text)
        result = predict(features)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
