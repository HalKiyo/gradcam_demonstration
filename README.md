# Grad-CAM demonstration

海面水温偏差 (3か月分) からタイ北部の8月平均降水量を予測するCNNを作り，Grad-CAMで予測根拠を可視化するデモです。

## ディレクトリ構成

```
.
├── 20221124_gradcam.ipynb   発表スライド (RISE / nbconvert slides)
├── gradcam_demo/            ロジック本体
│   ├── __init__.py          パスと定数
│   ├── data.py              データ読み込み, 欠損値マスク
│   ├── model.py             CNNの定義, 学習済み重みの読み込み
│   ├── gradcam.py           Grad-CAM (tf.GradientTape)
│   └── plot.py              地図描画
├── scripts/
│   ├── train.py             訓練して重みを保存
│   └── run_gradcam.py       Grad-CAMを計算して図を保存
├── assets/                  スライド用画像
├── ckpt/weights.h5          学習済み重み
└── data/inputdata.npz       入力データ
```

## 使い方

```bash
pip install -r requirements.txt

# 訓練 (ckpt/weights.h5 を上書き)
python scripts/train.py --epochs 10

# Grad-CAM の図を figures/ に保存
python scripts/run_gradcam.py --index 44
```

スライドは `jupyter nbconvert --to slides 20221124_gradcam.ipynb` で書き出せます。
