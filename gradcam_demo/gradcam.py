import numpy as np
import tensorflow as tf

from .model import LAST_CONV_LAYER


def _grad_model(model, layer_name):
    """入力から (指定したconv層の出力, 予測値) を返すモデルを作る."""
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    h = inputs
    for layer in model.layers:
        h = layer(h)
        if layer.name == layer_name:
            conv_output = h
    return tf.keras.Model(inputs, [conv_output, h])


def grad_cam(model, x, y, layer_name=LAST_CONV_LAYER):
    """各サンプルの予測誤差に対するGrad-CAMヒートマップを返す.

    x: (N, lat, lon, ch) の入力, y: (N,) の正解値.
    返り値は (N, lat, lon) で, サンプルごとに 0-1 に正規化されている.
    """
    x = tf.convert_to_tensor(np.asarray(x), dtype=tf.float32)
    y = tf.convert_to_tensor(np.asarray(y), dtype=tf.float32)
    grad_model = _grad_model(model, layer_name)

    # ---1. 予測のLossを計算---
    # サンプルごとの勾配が互いに独立になるよう sum をとる
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(x)
        loss = tf.reduce_sum(tf.square(pred[:, 0] - y))
    # ---2. 予測のLossから最後のconv層への勾配を計算---
    grads = tape.gradient(loss, conv_output)
    # ---3. チャンネルごとに勾配の平均を計算して，各チャンネルの重要度とする---
    weights = tf.reduce_mean(grads, axis=(1, 2))
    # ---4. conv層の出力にチャンネルごとの重みをかけて，足し合わせる---
    cam = tf.einsum("nhwc,nc->nhw", conv_output, weights)
    cam = tf.image.resize(cam[..., tf.newaxis], x.shape[1:3], method="bilinear")[..., 0]
    cam = tf.nn.relu(cam).numpy()

    cam_min = cam.min(axis=(1, 2), keepdims=True)
    cam_max = cam.max(axis=(1, 2), keepdims=True)
    return (cam - cam_min) / (cam_max - cam_min + 1e-12)
