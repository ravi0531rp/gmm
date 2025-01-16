from itertools import cycle
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot


def scatter_plot(
    transformed_data: pd.DataFrame,
    color: str = "y",
    size: float = 2.0,
    splot: Optional[pyplot.Axes] = None,
    label: Optional[List[str]] = None,
):
    """Generates a 2D scatter plot."""
    if splot is None:
        splot = pyplot.subplot()
    columns = transformed_data.columns
    splot.scatter(
        transformed_data.loc[:, columns[0]],
        transformed_data.loc[:, columns[1]],
        s=size,
        c=color,
        label=label,
        alpha=0.7,
        edgecolors='w',
    )
    splot.set_aspect("equal", "box")
    splot.set_xlabel("1st Component")
    splot.set_ylabel("2nd Component")
    if label:
        splot.legend()

def plot_density_estimation_results(
    X: pd.DataFrame,
    Y_: np.ndarray,
    
    means: np.ndarray,
    covariances: np.ndarray,
    title: str,
):
    """Plots Gaussian Mixture density estimation results."""
    color_iter = cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "g"])
    pyplot.figure(figsize=(8, 6))
    splot = pyplot.subplot()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        scatter_plot(X.loc[Y_ == i], color=color, splot=splot, label=f"Cluster {i}")
        angle = np.arctan2(u[1], u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    pyplot.title(title)
    pyplot.xlabel("1st Component")
    pyplot.ylabel("2nd Component")
    pyplot.show()


def plot_finnish_parties(transformed_data: pd.DataFrame, splot: Optional[pyplot.Axes] = None):
    """Plots Finnish political parties on a 2D scatter plot."""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]

    if splot is None:
        _, splot = pyplot.subplots(figsize=(4, 3))

    for group in finnish_parties:
        # Adjusting to filter using the correct index levels by position
        subset = transformed_data[
            (transformed_data.index.get_level_values(2) == group["country"]) &  # 2 is for 'country'
            (transformed_data.index.get_level_values(1).isin(group["parties"]))  # 1 is for 'party'
        ]
        scatter_plot(
            subset[[subset.columns[0], subset.columns[1]]],
            color=group["color"],
            splot=splot,
            label=", ".join(group["parties"]),
            size = 20
        )
    splot.set_title("Finnish Political Parties in 2D Space")
    splot.set_xlabel("1st Component")
    splot.set_ylabel("2nd Component")
    splot.legend()
    pyplot.show()


