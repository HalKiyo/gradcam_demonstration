import numpy as np
import numpy.ma as ma

from . import DATA_PATH, MISSING_THRESHOLD


def load_data(path=DATA_PATH):
    """訓練データと検証データを (x_train, y_train, x_val, y_val) で返す."""
    with np.load(path) as npz:
        return npz["x_train"], npz["y_train"], npz["x_val"], npz["y_val"]


def mask_missing(x):
    """描画用に欠損値をマスクする."""
    return ma.masked_where(x > MISSING_THRESHOLD, x)
