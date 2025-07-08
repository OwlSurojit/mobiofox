import dask.array as da
import matplotlib.pyplot as plt
import numpy as np


def show_histogram(data: np.ndarray, layer_name: str = ""):
    if isinstance(data, da.Array):
        data = data.compute()
    if data.ndim > 1:
        data = data.ravel()
    if np.issubdtype(data.dtype, np.integer):
        min_, max_ = data.min(), data.max()
        hist = np.bincount(data)[min_:max_]
        bin_edges = np.arange(min_, max_)
    else:
        hist, bin_edges = np.histogram(data, bins=1000)
        bin_edges = bin_edges[:-1]

    fig, ax = plt.subplots()
    ax.plot(bin_edges, hist)
    ax.fill_between(bin_edges, hist)
    ax.set_yscale("log")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {layer_name}" if layer_name else "Histogram")
    fig.show()
