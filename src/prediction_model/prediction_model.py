import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.constants import base_path
from src.functions import (
    get_image_paths_and_labels,
    clean_labels,
    prepare_dataset,
)
from .edit_distance_callback import EditDistanceCallback
from .model import build_model

np.random.seed(42)
tf.random.set_seed(42)


def prediction_model():
    # Dataset splitting
    words_list = []
    words = open(f"{base_path}/words.txt", "r").readlines()
    for line in words:
        if line[0] == "#":
            continue
        if line.split(" ")[1] != "err":  # We don't need to deal with erred entries.
            words_list.append(line)
    np.random.shuffle(words_list)

    split_idx = int(0.8 * len(words_list))
    train_samples = words_list[:split_idx]
    test_samples = words_list[split_idx:]

    val_split_idx = int(0.5 * len(test_samples))
    validation_samples = test_samples[:val_split_idx]
    test_samples = test_samples[val_split_idx:]

    assert len(words_list) == len(train_samples) + len(validation_samples) + len(
        test_samples
    )

    # print(f"Total training samples: {len(train_samples)}")
    # print(f"Total validation samples: {len(validation_samples)}")
    # print(f"Total test samples: {len(test_samples)}")

    # Data input pipeline
    train_img_paths, train_labels = get_image_paths_and_labels(samples=train_samples)
    validation_img_paths, validation_labels = get_image_paths_and_labels(
        samples=validation_samples
    )
    test_img_paths, test_labels = get_image_paths_and_labels(samples=test_samples)

    # Find maximum length and the size of the vocabulary in the training data.
    train_labels_cleaned = []
    characters = set()
    max_len = 0
    for label in train_labels:
        label = label.split(" ")[-1].strip()
        for char in label:
            characters.add(char)
        max_len = max(max_len, len(label))
        train_labels_cleaned.append(label)
    characters = sorted(list(characters))

    # print("Maximum length: ", max_len)
    # print("Vocab size: ", len(characters))

    # Check some label samples.
    # print(train_labels_cleaned[:10])

    validation_labels_cleaned = clean_labels(labels=validation_labels)
    test_labels_cleaned = clean_labels(labels=test_labels)

    # Prepare tf.data.Dataset objects
    train_ds = prepare_dataset(
        image_paths=train_img_paths,
        labels=train_labels_cleaned,
        characters=characters,
        max_len=max_len,
    )
    validation_ds = prepare_dataset(
        image_paths=validation_img_paths,
        labels=validation_labels_cleaned,
        characters=characters,
        max_len=max_len,
    )
    test_ds = prepare_dataset(
        image_paths=test_img_paths,
        labels=test_labels_cleaned,
        characters=characters,
        max_len=max_len,
    )

    validation_images = []
    validation_labels = []
    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    # Training
    epochs = 1  # To get good results this should be at least 50.
    model = build_model(characters)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(
        prediction_model, validation_images, validation_labels, max_len
    )

    # Train the prediction_model.
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    return prediction_model, test_ds, characters, max_len
