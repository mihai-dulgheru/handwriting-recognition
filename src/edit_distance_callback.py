import numpy as np
from tensorflow import keras

from functions import calculate_edit_distance


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(self, prediction_model, validation_images, validation_labels, max_len):
        super().__init__()
        self.prediction_model = prediction_model
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in range(len(self.validation_images)):
            labels = self.validation_labels[i]
            predictions = self.prediction_model.predict(self.validation_images[i])
            edit_distances.append(calculate_edit_distance(labels, predictions, self.max_len).numpy())

        print(f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}")
