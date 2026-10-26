import cartopy.crs as ccrs
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

IMG_EXTENT = (-180, 180, -55, 65)
SST_NORM = colors.Normalize(vmin=-3, vmax=3)


def _map_axes(fig, *args):
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = fig.add_subplot(*args, projection=proj)
    ax.coastlines(resolution="50m", lw=0.5)
    ax.gridlines(xlocs=mticker.MultipleLocator(90),
                 ylocs=mticker.MultipleLocator(45),
                 linestyle="-", color="gray")
    return ax, proj


def plot_inputs(x):
    """1サンプル分の入力 (lat, lon, ch) をチャンネルごとに描く."""
    n_ch = x.shape[-1]
    fig, axes = plt.subplots(n_ch, 1, tight_layout=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        img = ax.imshow(x[:, :, i], cmap="RdBu_r", norm=SST_NORM)
        fig.colorbar(img, ax=ax)
    return fig


def plot_heatmap(heatmap):
    fig = plt.figure()
    ax, proj = _map_axes(fig, 1, 1, 1)
    mat = ax.matshow(heatmap, cmap="BuPu", extent=IMG_EXTENT, transform=proj)
    fig.colorbar(mat, ax=ax, orientation="horizontal")
    return fig


def plot_input_and_heatmap(img, heatmap):
    """上段にヒートマップ, 下段に入力を描く."""
    fig = plt.figure()
    ax, proj = _map_axes(fig, 2, 1, 1)
    mat = ax.matshow(heatmap, cmap="BuPu", extent=IMG_EXTENT, transform=proj)
    fig.colorbar(mat, ax=ax, orientation="horizontal")

    ax = fig.add_subplot(2, 1, 2, projection=proj)
    mat = ax.matshow(img, cmap="RdBu_r", norm=SST_NORM)
    fig.colorbar(mat, ax=ax, orientation="horizontal")
    return fig


def plot_prediction(pred, y):
    """予測値と正解値の散布図. 凡例は相関係数."""
    fig, ax = plt.subplots()
    corr = np.corrcoef(pred, y)[0, 1]
    ax.scatter(pred, y, label=f"r = {corr:.3f}")
    line = np.linspace(-1.5, 3)
    ax.plot(line, line)
    ax.set_xlabel("prediction")
    ax.set_ylabel("observation")
    ax.legend()
    return fig
