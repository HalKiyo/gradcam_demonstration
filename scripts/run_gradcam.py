"""訓練済みモデルでGrad-CAMを計算して図を保存する.

    python scripts/run_gradcam.py --index 44 --outdir figures
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from gradcam_demo import WEIGHTS_PATH
from gradcam_demo.data import load_data, mask_missing
from gradcam_demo.gradcam import grad_cam
from gradcam_demo.model import load_trained_model
from gradcam_demo.plot import plot_heatmap, plot_input_and_heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=44, help="可視化する検証サンプル番号")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    _, _, x_val, y_val = load_data()
    model = load_trained_model(args.weights)
    heatmaps = grad_cam(model, x_val, y_val)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig = plot_input_and_heatmap(mask_missing(x_val[args.index, :, :, 0]),
                                 heatmaps[args.index])
    fig.savefig(args.outdir / f"gradcam_{args.index}.png", dpi=150)
    fig = plot_heatmap(heatmaps.mean(axis=0))
    fig.savefig(args.outdir / "gradcam_mean.png", dpi=150)
    print(f"saved figures to {args.outdir}")


if __name__ == "__main__":
    main()
