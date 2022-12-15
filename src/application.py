# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
#
# from functions import decode_batch_predictions
# from prediction_model import get_prediction_model
import os

from flask import Flask, render_template, request

IMAGES_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER


# This code will be executed while creating the Flask application.
# prediction_model, test_ds, characters, max_len = get_prediction_model()


def predict(img):
    return "MOVE"


@app.route('/', methods=['POST', 'GET'])
def hello():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            dst = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            prediction = predict(image)
            image.save(dst)
            return render_template('index.html', image=dst, prediction=prediction)
    return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_image():
#     image = request.files['image']
#
#     # Read the contents of the file into a string
#     file_contents = image.read()
#
#     # Convert the string to a EagerTensor
#     tensor = tf.convert_to_tensor(file_contents)
#
#     predictions = prediction_model.predict(tensor)
#     prediction_texts = decode_batch_predictions(predictions=predictions, characters=characters, max_len=max_len)
#
#     if prediction_texts:
#         return prediction_texts[0]
#     else:
#         return "No predictions available"


# @app.route("/check-results")
# def check_results():
#     for batch in test_ds.take(1):
#         batch_images = batch["image"]
#         _, ax = plt.subplots(4, 4, figsize=(15, 8))
#         predictions = prediction_model.predict(batch_images)
#         prediction_texts = decode_batch_predictions(predictions=predictions, characters=characters, max_len=max_len)
#         for i in range(16):
#             img = batch_images[i]
#             img = tf.image.flip_left_right(img)
#             img = tf.transpose(img, perm=[1, 0, 2])
#             img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
#             img = img[:, :, 0]
#             title = f"Prediction: {prediction_texts[i]}"
#             ax[i // 4, i % 4].imshow(img, cmap="gray")
#             ax[i // 4, i % 4].set_title(title)
#             ax[i // 4, i % 4].axis("off")
#     plt.show()
#     return ''


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
