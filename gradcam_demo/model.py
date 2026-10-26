import tensorflow as tf
from tensorflow.keras import layers, models

from . import INPUT_SHAPE, WEIGHTS_PATH

# Grad-CAM で使う最後の畳み込み層
LAST_CONV_LAYER = "conv3"


def build_model(learning_rate=1e-4):
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(30, (4, 8), activation="tanh", padding="same", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Conv2D(30, (2, 4), activation="tanh", padding="same", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Conv2D(30, (2, 4), activation="tanh", padding="same", name=LAST_CONV_LAYER),
        layers.Flatten(name="flatten"),
        layers.Dense(30, activation="tanh", name="dense"),
        layers.Dense(1, activation="linear", name="output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def load_trained_model(path=WEIGHTS_PATH):
    model = build_model()
    model.load_weights(path)
    return model
