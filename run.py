import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from app.model_utils import extract_features_from_text, predict
import fitz  # PyMuPDF
import docx

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        doc = fitz.open(stream=file_stream, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print("PDF extraction error:", e)
    return text.strip()

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        print("DOCX extraction error:", e)
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    extracted_text = ""

    if request.method == 'POST':
        # 1. First check raw text input
        text_input = request.form.get('text', '').strip()
        uploaded_file = request.files.get('file')

        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext == '.pdf':
                extracted_text = extract_text_from_pdf(uploaded_file.read())
            elif file_ext == '.docx':
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(path)
                extracted_text = extract_text_from_docx(path)
                os.remove(path)
            elif file_ext == '.txt':
                extracted_text = uploaded_file.read().decode('utf-8').strip()

        if not extracted_text:
            extracted_text = text_input

        if extracted_text:
            features = extract_features_from_text(extracted_text)
            result, confidence = predict(features)
        else:
            result = "No text found for prediction."

    return render_template('index.html', result=result, confidence=confidence, extracted_text=extracted_text)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
