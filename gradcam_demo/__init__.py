"""海面水温偏差からタイ北部の降水量を予測するCNNと、そのGrad-CAM可視化."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "inputdata.npz"
WEIGHTS_PATH = ROOT / "ckpt" / "weights.h5"

# 入力データの形状 (緯度, 経度, 月数)
INPUT_SHAPE = (24, 72, 3)
# 陸地などの欠損値は 99999 で埋められている
MISSING_THRESHOLD = 9999
