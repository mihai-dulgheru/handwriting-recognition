import os

import tensorflow as tf

base_path = "../data"
base_image_path = os.path.join(base_path, "words")
batch_size = 64
padding_token = 99
image_width = 128
image_height = 32
AUTOTUNE = tf.data.AUTOTUNE
