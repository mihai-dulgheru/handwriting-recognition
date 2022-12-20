import os

from flask import Flask, render_template, request

from functions import predict
from model import prediction_model

IMAGES_FOLDER = os.path.join("static", "images")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = IMAGES_FOLDER

# This code will be executed while creating the Flask application.
prediction_model, test_ds, characters, max_len = prediction_model()


@app.route("/", methods=["POST", "GET"])
def hello():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            dst = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_image.png")
            image.save(dst)
            prediction = predict(
                image_path=dst,
                prediction_model=prediction_model,
                characters=characters,
                max_len=max_len,
            )
            return render_template("index.html", image=dst, prediction=prediction)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
