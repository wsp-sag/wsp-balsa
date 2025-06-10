from __future__ import annotations

from os import PathLike
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


def convergence_boxplot(
    targets: pd.DataFrame,
    results: pd.DataFrame,
    filter_func: Callable[[pd.Series], pd.Series],
    *,
    adjust_target: bool = True,
    percentage: bool = True,
    band: Tuple[float, float] = None,
    simple_labels: bool = True,
    ax: Axes = None,
    fp: Union[str, PathLike] = None,
    title: str = None,
) -> Axes:
    """Measures convergence of constrained location-choice models (such as work-location choice). Can be used to
    produce multiple box plots for different sub-sets of zones, usually based on size.

    Args:
        targets (pandas.DataFrame):
        results (pandas.DataFrame):
        filter_func (Callable[[pandas.Series], pandas.Series]):
        adjust_target (bool, optional):
        percentage (bool, optional):
        band (Tuple[float, float], optional):
        simple_labels (bool, optional):
        ax (Axes, optional):
        fp (str | PathLike, optional):
        title (str, optional):

    Returns:
        matplotlib.Axes
    """

    assert results.columns.equals(targets.columns)

    columns, filters, n = [], [], 0
    for colname in targets:
        filter_ = filter_func(targets[colname])
        filters.append(filter_)
        n = max(n, filter_.sum())
        columns.append(colname)

    unlabelled_zones = np.full([n, len(columns)], np.nan, dtype=np.float64)
    model_sums, target_sums = [], []

    for i, (colname, filter_) in enumerate(zip(columns, filters)):
        m = filter_.sum()

        model_vector = results.loc[filter_, colname]
        target_vector = targets.loc[filter_, colname]

        if adjust_target and target_vector.sum() > 0:
            factor = model_vector.sum() / target_vector.sum()
            target_vector = target_vector * factor

        model_sums.append(model_vector.sum())
        target_sums.append(target_vector.sum())

        err = model_vector - target_vector
        if percentage:
            err /= target_vector

        unlabelled_zones[:m, i] = err.values

    if not simple_labels:
        columns = [
            "{}\n{} workers\n{} jobs\n{} zones".format(c, model_sums[i], int(target_sums[i]), filters[i].sum())
            for i, c in enumerate(columns)
        ]
    unlabelled_zones = pd.DataFrame(unlabelled_zones, columns=columns)

    with np.errstate(invalid="ignore"):
        ax = unlabelled_zones.plot.box(ax=ax, figsize=[12, 6])
        ax.axhline(0)

        if percentage:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(np.round(x, 2) * 100)))
            ax.set_ylabel("Relative error ((Model - Target) / Target)")
        else:
            ax.set_ylabel("Error (Model - Target)")

        xlabel = "Occupation & Employment Status"
        if adjust_target:
            xlabel += "\n(Targets adjusted to match model totals)"
        ax.set_xlabel(xlabel)

        if band is not None:
            lower, upper = band
            ax.axhline(lower, color="black", linewidth=1, alpha=0.5)
            ax.axhline(upper, color="black", linewidth=1, alpha=0.5)

        if title:
            ax.set_title(title)

        if fp is not None:
            plt.savefig(str(fp))

        return ax


def location_summary(
    model: pd.DataFrame,
    target: pd.DataFrame,
    ensemble_names: pd.Series,
    *,
    title: str = "",
    fp: Union[str, PathLike] = None,
    dpi: int = 150,
    district_name: str = "Ensemble",
) -> Axes:
    """Creates a compound plot showing total attractions to specified locations

    Args:
        model (pandas.DataFrame):
        target (pandas.DataFrame):
        ensemble_names (pandas.Series):
        title (str, optional):
        fp (str | PathLike, optional):
        dpi (int, optional):
        district_name (str, optional):

    Returns:
        matplotlib.Axes
    """

    fig, ax = plt.subplots(1, 3, figsize=[16, 8], gridspec_kw={"width_ratios": [4, 2, 2]})

    model_col = model.reindex(ensemble_names.index, fill_value=0)
    target_col = target.reindex(ensemble_names.index, fill_value=0)

    factor = model_col.sum() / target_col.sum()
    target_col = target_col * factor

    df = pd.DataFrame({"Model": model_col, "Target": target_col})
    df.index = ensemble_names
    df.index.name = district_name
    sub_ax = ax[0]
    df.plot.barh(ax=sub_ax)
    sub_ax.set_title(title)
    sub_ax.invert_yaxis()

    short_labels = np.arange(1, len(ensemble_names) + 1)
    short_labels[-1] = 99

    sub_ax = ax[1]
    diff = model_col - target_col
    diff.index = short_labels

    colours = pd.Series("grey", index=diff.index)
    colours.loc[diff >= 2000] = "skyblue"
    colours.loc[diff >= 5000] = "dodgerblue"
    colours.loc[diff <= -2000] = "lightsalmon"
    colours.loc[diff <= -5000] = "tomato"

    diff.plot.barh(ax=sub_ax, color=colours)
    sub_ax.set_title("Error (Model - Target)")
    sub_ax.invert_yaxis()

    sub_ax = ax[2]
    perc_diff = diff / target_col
    perc_diff.index = short_labels

    colours = pd.Series("grey", index=perc_diff.index)
    colours.loc[perc_diff >= 0.1] = "skyblue"
    colours.loc[perc_diff >= 0.25] = "dodgerblue"
    colours.loc[perc_diff <= -0.1] = "lightsalmon"
    colours.loc[perc_diff <= -0.25] = "tomato"

    perc_diff.plot.barh(ax=sub_ax, color=colours)
    sub_ax.set_title(" % Error (Model - Target) / Target")
    sub_ax.invert_yaxis()

    plt.tight_layout()

    if fp is not None:
        fig.savefig(str(fp), dpi=dpi)

    return ax


def trumpet_diagram(
    counts: pd.Series,
    model_volume: pd.Series,
    *,
    categories: Union[pd.Series, List[pd.Series]] = None,
    category_colours: Dict[Union[str, tuple], str] = None,
    category_markers: Dict[Union[str, tuple], str] = None,
    label_format: str = None,
    title: str = "",
    y_bounds: Tuple[float, float] = (-2, 2),
    ax: Axes = None,
    x_label: str = "Count volume",
    legend: bool = True,
    **kwargs,
) -> Axes:
    """Plots an auto volumes "trumpet" diagram of relative error vs. target count, and will draw min/max error curves
    based on FHWA guidelines. Can be used to plot different categories of count locations.

    Args:
        counts (pandas.Series): Target counts. Each item represents a different count location. Index does not need to
            be unique.
        model_volume (pandas.Series): Modelled volumes for each location. The index must match the counts Series.
        categories (pandas.Series | List[pandas.Series], optional): Defaults to ``None``. Optional classification
            of each count location. Must match the index of the count Series. Can be provided as a List of Series (which
            all must match the count index) to enable tuple-based categorization.
        category_colours (Dict[str | tuple, str], optional): Defaults to ``None``. Mapping of each category to a
            colour, specified as a hex string. Only used when categories are provided. Missing categories revert to
            ``None``, using the default colour for the style.
        category_markers (Dict[str | tuple, str], optional): Defaults to ``None``. Mapping of each category to a
            matplotlib marker string (see https://matplotlib.org/api/markers_api.html for options). Only used when
            categories are provided. Missing categories revert to ``None``, using the default marker for the style.
        label_format (str, optional): Defaults to ``None``. Used to convert category values (especially tuples) into
            readable strings for the plot legend. The relevant line of code is
            ``current_label = label_format % category_key``. Only used when categories are provided.
        title (str, optional): The title to set on the plot.
        y_bounds (tuple, optional): Defaults to ``(-2, 2)``, or (-200%, 200%). Limit of the Y-Axis. This is needed
            because relative errors can be very high close to the y-intercept of the plot.
        ax (matplotlib.Axes, optional): Defaults to ``None``. Sub-Axes to add this plot to, if using ``subplots()``.
        x_label (str, optional): Defaults to ``'Count volume'``. Label to use for the X-axis. The Y-axis is always
            "Relative Error"
        legend (bool, optional): Defaults to ``True``. Flag to add a legend.
        **kwargs: Additional kwargs to pass to ``DataFrame.plot.scatter()``

    Returns:
        matplotlib.Axes:
            The Axes object generated from the plot. For most use cases, this is not really needed.
    """

    assert model_volume.index.equals(counts.index)

    n_categories = 0
    if categories is not None:
        if isinstance(categories, list):
            for s in categories:
                assert s.index.equals(model_volume.index)
            if label_format is None:
                label_format = "-".join(["%s"] * len(categories))
            categories = pd.MultiIndex.from_arrays(categories)
            n_categories = len(categories.unique())
        else:
            assert categories.index.equals(model_volume.index)
            n_categories = categories.nunique()

        if category_colours is None:
            category_colours = {}
        if category_markers is None:
            category_markers = {}
    if label_format is None:
        label_format = "%s"

    df = pd.DataFrame({"Model Volume": model_volume, "Count Volume": counts})
    df["Error"] = df["Model Volume"] - df["Count Volume"]
    df["% Error"] = df["Error"] / df["Count Volume"]

    if n_categories > 1:
        for category_key, subset in df.groupby(categories):
            current_label = label_format % category_key
            current_color = category_colours[category_key] if category_key in category_colours else None
            current_marker = category_markers[category_key] if category_key in category_markers else None

            ax = subset.plot.scatter(
                x="Count Volume",
                y="% Error",
                ax=ax,
                c=current_color,
                marker=current_marker,
                label=current_label,
                **kwargs,
            )
    else:
        ax = df.plot.scatter(x="Count Volume", y="% Error", ax=ax, **kwargs)

    top = counts.max() * 1.05  # Add 5% to the top, to give some visual room on the right side
    xs = np.arange(1, top, 10)
    pos_ys = (-13.7722 + (555.1382 * xs**-0.26025)) / 100.0
    neg_ys = pos_ys * -1

    ax.plot(xs, np.zeros(len(xs)), color="black")
    ax.plot(xs, pos_ys, color="red", linewidth=1, zorder=1)
    ax.plot(xs, neg_ys, color="red", linewidth=1, zorder=1)

    ax.set_xlim(0, top)

    bottom, top = y_bounds
    ax.set_ylim(bottom, top)
    ax.set_yticks(np.arange(bottom, top, 0.25))

    ax.set_title(title)
    ax.set_ylabel("Relative Error")
    ax.set_xlabel(x_label)
    if legend:
        ax.legend()

    return ax
