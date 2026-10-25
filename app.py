from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from detector import analyze_media

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    if "media" not in request.files:
        return "No file selected"

    file = request.files["media"]

    if file.filename == "":
        return "Please select a file"

    filename = secure_filename(file.filename)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(file_path)

    result = analyze_media(file_path)

    return render_template(
        "result.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
