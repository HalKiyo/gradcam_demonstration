"""CNNを訓練して重みを保存する.

    python scripts/train.py --epochs 10 --output ckpt/weights.h5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gradcam_demo import WEIGHTS_PATH
from gradcam_demo.data import load_data
from gradcam_demo.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    args = parser.parse_args()

    x_train, y_train, x_val, y_val = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=args.epochs,
              batch_size=args.batch_size, verbose=2)

    pred_val = model.predict(x_val, verbose=0)[:, 0]
    print(f"validation correlation: {np.corrcoef(pred_val, y_val)[0, 1]:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(args.output)
    print(f"saved weights to {args.output}")


if __name__ == "__main__":
    main()
