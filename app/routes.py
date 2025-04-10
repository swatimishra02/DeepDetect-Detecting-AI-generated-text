from flask import Blueprint, render_template, request
from app.model_utils import predict_text

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        input_text = request.form["input_text"]
        result = predict_text(input_text)
    return render_template("index.html", result=result)
