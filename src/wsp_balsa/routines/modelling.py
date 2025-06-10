from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numexpr as ne
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def tlfd(
    values: ArrayLike,
    *,
    bin_start: int = 0,
    bin_end: int = 200,
    bin_step: int = 2,
    weights: ArrayLike = None,
    intrazonal: ArrayLike = None,
    label_type: str = "MULTI",
    include_top: bool = False,
) -> pd.Series:
    """Generates a Trip Length Frequency Distribution (i.e. a histogram) from given data. Produces a "pretty" Pandas
    object suitable for charting.

    Args:
        values (ArrayLike): A vector of trip lengths, with a length  of "N". Can be provided from a table of trips, or
            from a matrix (in "tall" format).
        bin_start (int, optional): Defaults is ``0``. The minimum bin value, in the same units as ``values``.
        bin_end (int, optional): Defaults to ``200``. The maximum bin value, in the same units as ``values``. Values
            over this limit are either ignored, or counted under a separate category (see ``include_top``)
        bin_step (int, optional): Default is ``2``. The size of each bin, in the same unit as ``values``.
        weights (ArrayLike, optional): Defaults to ``None``. A vector of weights to use of length "N", to produce a
            weighted histogram.
        intrazonal (ArrayLike, optional): Defaults to ``None``. A boolean vector indicating which values are considered
            "intrazonal". When specified, prepends an ``intrazonal`` category to the front of the histogram.
        label_type (str, optional): Defaults to ``'MULTI'``. The format of the returned index. Options are:
            - ``MULTI``: The returned index will be a 2-level MultiIndex ['from', 'to'];
            - ``TEXT``: The returned index will be text-based: "0 to 2";
            - ``BOTTOM``: The returned index will be the bottom of each bin; and
            - ``TOP``: The returned index will be the top of each bin.
        include_top (bool, optional): Defaults to ``False``. If True, the function will count all values (and weights,
            if provided) above the `bin_top`, and add them to the returned Series. This bin is described as going from
            `bin_top` to `inf`.

    Returns:
        pandas.Series:
            The weighted or unweighted histogram, depending on the options configured above.

    """
    bins = list(range(bin_start, bin_end + bin_step, bin_step))

    iz_total = None
    if intrazonal is not None:
        if weights is not None:
            iz_total = weights.loc[intrazonal].sum()
            weights = weights.loc[~intrazonal]
        else:
            iz_total = intrazonal.sum()

        values = values.loc[~intrazonal]

    if weights is not None:
        hist, _ = np.histogram(values, weights=weights, bins=bins)
    else:
        hist, _ = np.histogram(values, bins=bins)

    new_len = len(hist)
    if intrazonal is not None:
        new_len += 1
    if include_top:
        new_len += 1
    new_hist = np.zeros(shape=new_len, dtype=hist.dtype)
    lower_index = 0
    upper_index = new_len

    if intrazonal is not None:
        new_hist[0] = iz_total
        bins.insert(0, "intrazonal")
        lower_index += 1
    if include_top:
        if weights is not None:
            top = weights.loc[values >= bin_end].sum()
        else:
            top = (values >= bin_end).sum()

        new_hist[-1] = top
        bins.append(np.inf)
        upper_index -= 1
    new_hist[lower_index:upper_index] = hist

    label_type = label_type.upper()
    if label_type == "MULTI":
        index = pd.MultiIndex.from_arrays([bins[:-1], bins[1:]], names=["from", "to"])
    elif label_type == "TOP":
        index = pd.Index(bins[1:])
    elif label_type == "BOTTOM":
        index = pd.Index(bins[:-1])
    elif label_type == "TEXT":
        s0 = pd.Series(bins[:-1]).astype(str)
        s1 = pd.Series(bins[1:]).astype(str)
        index = pd.Index(s0.str.cat(s1, sep=" to "))
    else:
        raise NotImplementedError(label_type)

    new_hist = pd.Series(new_hist, index=index)

    return new_hist


def _get_distance_equation(method: str) -> str:
    if method.lower() == "euclidean":
        expr = "sqrt((x0 - x1)**2 + (y0 - y1) ** 2) * coord_unit"
    elif method.lower() == "manhattan":
        expr = "(abs(x0 - x1) + abs(y0 - y1)) * coord_unit"
    elif method.lower() == "haversine":
        y0 = "(y0 * pi / 180)"
        y1 = "(y1 * pi / 180)"
        delta_lon = "((x1 - x0) * pi / 180)"
        delta_lat = "((y1 - y0) * pi / 180)"
        part1 = "sin({delta_lat} / 2.0)**2 + cos({y0}) * cos({y1}) * (sin({delta_lon} / 2.0))**2".format(
            delta_lat=delta_lat, delta_lon=delta_lon, y0=y0, y1=y1
        )
        expr = "6371.0 * earth_radius_factor* 2.0 * arctan2(sqrt({part1}), sqrt(1.0 - {part1})) * coord_unit".format(
            part1=part1
        )
    else:
        raise NotImplementedError(method.lower())
    return expr


def _prepare_distance_kwargs(kwargs: Dict[str, Any]):
    defaults = {"coord_unit": 1.0, "earth_radius_factor": 1.0, "pi": np.pi}
    for key, val in defaults.items():
        if key not in kwargs:
            kwargs[key] = val


def _check_vectors(description: str, *vectors):
    if len(vectors) < 1:
        return []

    first = vectors[0]
    retval = []
    common_index, common_length = None, None
    if isinstance(first, pd.Series):
        common_index = first.index
        common_length = len(common_index)
        retval.append(first.values[...])
    else:
        retval.append(first[...])

    for vector in vectors[1:]:
        if isinstance(vector, pd.Series):
            assert vector.index.equals(common_index), "All %s Series must have the same index" % description
            retval.append(vector.values[...])
        else:
            assert len(vector) == common_length, "All %s vectors must have the same length" % description
            retval.append(vector[...])

    return common_index, retval


def distance_matrix(
    x0: ArrayLike,
    y0: ArrayLike,
    *,
    labels0: ArrayLike = None,
    tall: bool = False,
    x1: ArrayLike = None,
    y1: ArrayLike = None,
    labels1: ArrayLike = None,
    method: str = "EUCLIDEAN",
    **kwargs,
) -> Union[pd.Series, pd.DataFrame, NDArray]:
    """Fastest method of computing a distance matrix from vectors of coordinates, using the NumExpr package. Supports
    several equations for computing distances.

    Accepts two or four vectors of x-y coordinates. If only two vectors are provided (x0, y0), the result will be the
    2D product of this vector with itself (vector0 * vector0). If all four are provided (x0, y0, x1, y1), the result
    will be the 2D product of the first and second vector (vector0 * vector1).

    Args:
        x0 (ArrayLike): Vector of x-coordinates, of length N0. Can be a Series to specify labels.
        y0 (ArrayLike): Vector of y-coordinates, of length N0. Can be a Series to specify labels.
        labels0 (ArrayLike, optional): Defaults to ``None``. Override set of labels to use if x0 and y0 are both raw
            Numpy arrays
        x1 (ArrayLike, optional): Defaults to ``None``. A second vector of x-coordinates, of length N1. Can be a Series
            to specify labels
        y1 (ArrayLike, optional): Defaults to ``None``. A second vector of y-coordinates, of length N1. Can be a Series
            to specify labels
        labels1 (ArrayLike): Override set of labels to use if x1 and y1 are both raw Numpy arrays
        tall (bool, optional): Defaults to ``False``. If True, returns a vector whose shape is N0 x N1. Otherwise,
            returns a matrix whose shape is (N0, N1).
        method (str, optional): Defaults to ``'EUCLIDEAN'``. Specifies the method by which to compute distance. Valid
            options are:
            ``'EUCLIDEAN'``: Computes straight-line, 'as-the-crow flies' distance.
            ``'MANHATTAN'``: Computes the Manhattan distance
            ``'HAVERSINE'``: Computes distance based on lon/lat.
        **kwargs: Additional scalars to pass into the evaluation context

    Kwargs:
        coord_unit (float):
            Factor applies directly to the result, defaulting to 1.0 (no conversion). Useful when the coordinates are
            provided in one unit (e.g. m) and the desired result is in a different unit (e.g. km). Only used for
            Euclidean or Manhattan distance
        earth_radius_factor (float):
            Factor to convert from km to other units when using Haversine distance

    Returns:
        pandas.Series, pandas.DataFrame or NDArray:
            A *Series* will be returned when ``tall=True``, and labels can be inferred and will always have 2-level
            MultiIndex. A *DataFrame* will be returned when ``tall=False`` and labels can be inferred. A *ndarray* will
            be returned when labels could not be inferred; if ``tall=True`` the array will be 1-dimensional, with shape
            (N x N,). Otherwise, it will 2-dimensional with shape (N, N)

    Note:
        The type of the returned object depends on whether labels can be inferred from the arguments. This is always
        true when the `labels` argument is specified, and the returned value will use cross-product of the `labels`
        vector.

        Otherwise, the function will try and infer the labels from the `x` and `y` objects, if one or both of them are
        provided as Series.
    """

    second_coords = x1 is not None and y1 is not None

    descr = "first coordinate" if second_coords else "coordinate"
    temp_labels, (x_array0, y_array0) = _check_vectors(descr, x0, y0)
    if labels0 is None:
        labels0 = temp_labels

    if second_coords:
        temp_labels, (x_array1, y_array1) = _check_vectors("second coordinate", x1, y1)
        if labels1 is None:
            labels1 = temp_labels
    else:
        x_array1 = x_array0[...]
        y_array1 = y_array0[...]
        labels1 = labels0

    n0, n1 = len(x_array0), len(x_array1)

    x_array0.shape = n0, 1
    y_array0.shape = n0, 1
    x_array1.shape = 1, n1
    y_array1.shape = 1, n1

    expr = _get_distance_equation(method)
    kwargs = kwargs.copy()
    _prepare_distance_kwargs(kwargs)
    kwargs["x0"] = x_array0
    kwargs["x1"] = x_array1
    kwargs["y0"] = y_array0
    kwargs["y1"] = y_array1

    raw_matrix = ne.evaluate(expr, local_dict=kwargs)
    labelled_result = labels0 is not None and labels1 is not None

    if tall:
        raw_matrix.shape = n0 * n1
        if not labelled_result:
            return raw_matrix

        mi = pd.MultiIndex.from_product([labels0, labels1])
        return pd.Series(raw_matrix, index=mi)
    elif not labelled_result:
        return raw_matrix

    return pd.DataFrame(raw_matrix, index=labels0, columns=labels1)


def distance_array(
    x0: ArrayLike,
    y0: ArrayLike,
    x1: ArrayLike,
    y1: ArrayLike,
    *,
    method: str = "euclidean",
    **kwargs,
) -> Union[NDArray, pd.Series]:
    """
    Fast method to compute distance between 2 (x, y) points, represented by 4 separate arrays, using the NumExpr
    package. Supports several equations for computing distances

    Args:
        x0 (ArrayLike): X or Lon coordinate of first point
        y0 (ArrayLike): Y or Lat coordinate of first point
        x1 (ArrayLike): X or Lon coordinate of second point
        y1 (ArrayLike): Y or Lat coordinate of second point
        method (str, optional): Defaults to ``'EUCLIDEAN'``. Specifies the method by which to compute distance. Valid
            options are:
            ``'EUCLIDEAN'``: Computes straight-line, 'as-the-crow flies' distance.
            ``'MANHATTAN'``: Computes the Manhattan distance
            ``'HAVERSINE'``: Computes distance based on lon/lat.
        **kwargs: Additional scalars to pass into the evaluation context

    Kwargs:
        coord_unit (float):
            Factor applies directly to the result, defaulting to 1.0 (no conversion). Useful when the
            coordinates are provided in one unit (e.g. m) and the desired result is in a different unit (e.g. km).
            Only used for Euclidean or Manhattan distance
        earth_radius_factor (float):
            Factor to convert from km to other units when using Haversine distance

    Returns:
        NDArray or pandas.Series:
            Distance from the vectors of first points to the vectors of second points. A Series is returned when one or
            more coordinate arrays are given as a Series object
    """

    labels, (x0, y0, x1, y1) = _check_vectors("coordinate", x0, y0, x1, y1)

    expr = _get_distance_equation(method)
    kwargs = kwargs.copy()
    _prepare_distance_kwargs(kwargs)
    kwargs["x0"] = x0
    kwargs["x1"] = x1
    kwargs["y0"] = y0
    kwargs["y1"] = y1

    result_array = ne.evaluate(expr, local_dict=kwargs)

    if labels is not None:
        return pd.Series(result_array, index=labels)

    return result_array


def indexers_for_map_matrix(
    row_labels: pd.Index,
    col_labels: pd.Index,
    superset: pd.Index,
    check: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if check:
        assert np.all(row_labels.isin(superset))
        assert np.all(col_labels.isin(superset))

    row_offsets = superset.get_indexer(row_labels)
    col_offsets = superset.get_indexer(col_labels)

    return row_offsets, col_offsets


def map_to_matrix(
    values: pd.Series,
    super_labels: pd.Index,
    fill_value: float = 0,
    row_col_labels: Tuple[pd.Series, pd.Series] = None,
    row_col_offsets: Tuple[NDArray, NDArray] = None,
    out: Union[pd.DataFrame, NDArray] = None,
    grouper_func: str = "sum",
    out_operand: str = "+",
) -> pd.DataFrame:

    # TODO: Check that `values` dtype is numeric, or at least, add-able

    if row_col_labels is not None:
        row_labels, col_labels = row_col_labels
        assert len(row_labels) == len(values)
        assert len(col_labels) == len(values)
    else:
        assert values.index.nlevels == 2
        row_labels = values.index.get_level_values(0)
        col_labels = values.index.get_level_values(1)

    if row_col_offsets is None:
        row_offsets, col_offsets = indexers_for_map_matrix(row_labels, col_labels, super_labels)
    else:
        row_offsets, col_offsets = row_col_offsets
        assert row_offsets.min() >= 0
        assert row_offsets.max() < len(super_labels)
        assert col_offsets.min() >= 0
        assert col_offsets.max() < len(super_labels)

    if out is not None:
        if isinstance(out, pd.DataFrame):
            assert out.index.equals(super_labels)
            assert out.columns.equals(super_labels)
            out = out.values  # Get the raw array from inside the frame
        elif isinstance(out, np.ndarray):
            nrows, ncols = out.shape
            assert nrows >= (row_offsets.max() - 1)
            assert ncols >= (col_offsets.max() - 1)
        else:
            raise TypeError(type(out))
        out_is_new = False
    else:
        out = np.full([len(super_labels)] * 2, fill_value=fill_value, dtype=values.dtype)
        out_is_new = True

    aggregated: pd.Series = values.groupby([row_offsets, col_offsets]).aggregate(func=grouper_func)
    xs, ys = aggregated.index.get_level_values(0), aggregated.index.get_level_values(1)
    if out_is_new:
        out[xs, ys] = aggregated.values
    else:
        out_name = "__OUT__"
        other_name = "__OTHER__"
        ld = {out_name: out[xs, ys], other_name: aggregated.values}
        ne.evaluate("{0} = {0} {1} {2}".format(out_name, out_operand, other_name), local_dict=ld)

    out = pd.DataFrame(out, index=super_labels, columns=super_labels)
    return out
