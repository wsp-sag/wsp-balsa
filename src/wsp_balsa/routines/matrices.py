from __future__ import annotations

from multiprocessing import cpu_count
from typing import Callable, Iterable, List, Tuple, Union
from warnings import warn

import numexpr as ne
import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from numba import njit, prange
except ImportError:

    def njit(*args, **kwargs):

        def decorator(func):
            return func

        return decorator

    prange = range

EPS = 1.0e-7


def matrix_balancing_1d(
    m: NDArray,
    a: NDArray,
    axis: int,
) -> NDArray:
    """Balances a matrix using a single constraint.

    Args:
        m (NDArray): The matrix (a 2-dimensional ndarray) to be balanced
        a (NDArray): The totals vector (a 1-dimensional ndarray) constraint
        axis (int): Direction to constrain (0 = along columns, 1 = along rows)

    Return:
        NDArray: A balanced matrix
    """

    assert axis in [0, 1], "axis must be either 0 or 1"
    assert m.ndim == 2, "`m` must be a two-dimensional matrix"
    assert a.ndim == 1, "`a` must be an one-dimensional vector"
    assert np.all(m.shape[axis] == a.shape[0]), "axis %d of matrices 'm' and 'a' must be the same." % axis

    return _balance(m, a, axis)


def matrix_balancing_2d(
    m: Union[NDArray, pd.DataFrame],
    a: NDArray,
    b: NDArray,
    *,
    totals_to_use: str = "raise",
    max_iterations: int = 1000,
    rel_error: float = 0.0001,
    n_threads: int = 1,
) -> Tuple[Union[NDArray, pd.DataFrame], float, int]:
    """Balances a two-dimensional matrix using iterative proportional fitting.

    Args:
        m (NDArray | pandas.DataFrame): The matrix (a 2-dimensional ndarray) to be balanced. If a DataFrame
            is supplied, the output will be returned as a DataFrame.
        a (NDArray): The row totals (a 1-dimensional ndarray) to use for balancing
        b (NDArray): The column totals (a 1-dimensional ndarray) to use for balancing
        totals_to_use (str, optional): Defaults to ``'raise'``. Describes how to scale the row and column totals if
            their sums do not match. Must be one of ['rows', 'columns', 'average', 'raise'].
            - rows: scales the columns totals so that their sums matches the row totals
            - columns: scales the row totals so that their sums matches the column totals
            - average: scales both row and column totals to the average value of their sums
            - raise: raises an Exception if the sums of the row and column totals do not match
        max_iterations (int, optional): Defaults to ``1000``. Maximum number of iterations
        rel_error (float, optional): Defaults to ``1.0E-4``. Relative error stopping criteria
        n_threads (int, optional): Defaults to ``1``. Number of processors for parallel computation. (Not used)

    Return:
        Tuple[NDArray | pandas.DataFrame, float, int]: The balanced matrix, residual, and n_iterations
    """
    max_iterations = int(max_iterations)
    n_threads = int(n_threads)

    # Test if matrix is Pandas DataFrame
    data_type = ""
    m_pd = None
    if isinstance(m, pd.DataFrame):
        data_type = "pd"
        m_pd = m
        m = m_pd.values

    if isinstance(a, pd.Series) or isinstance(a, pd.DataFrame):
        a = a.values
    if isinstance(b, pd.Series) or isinstance(b, pd.DataFrame):
        b = b.values

    # ##################################################################################
    # Validations:
    #   - m is an MxM square matrix, a and b are vectors of size M
    #   - totals_to_use is one of ['rows', 'columns', 'average']
    #   - the max_iterations is a +'ve integer
    #   - rel_error is a +'ve float between 0 and 1
    #   - the n_threads is a +'ve integer between 1 and the number of available processors
    # ##################################################################################
    valid_totals_to_use = ["rows", "columns", "average", "raise"]
    assert m.ndim == 2 and m.shape[0] == m.shape[1], "m must be a two-dimensional square matrix"
    assert (
        a.ndim == 1 and a.shape[0] == m.shape[0]
    ), "'a' must be a one-dimensional array, whose size matches that of 'm'"
    assert (
        b.ndim == 1 and b.shape[0] == m.shape[0]
    ), "'a' must be a one-dimensional array, whose size matches that of 'm'"
    assert totals_to_use in valid_totals_to_use, "totals_to_use must be one of %s" % valid_totals_to_use
    assert max_iterations >= 1, "max_iterations must be integer >= 1"
    assert 0 < rel_error < 1.0, "rel_error must be float between 0.0 and 1.0"
    assert 1 <= n_threads <= cpu_count(), (
        "n_threads must be integer between 1 and the number of processors (%d) " % cpu_count()
    )
    if n_threads > 1:
        raise NotImplementedError("Multiprocessing capability is not implemented yet.")

    # Scale row and column totals, if required
    a_sum = a.sum()
    b_sum = b.sum()
    if not np.isclose(a_sum, b_sum):
        if totals_to_use == "rows":
            b = np.multiply(b, a_sum / b_sum)
        elif totals_to_use == "columns":
            a = np.multiply(a, b_sum / a_sum)
        elif totals_to_use == "average":
            avg_sum = 0.5 * (a_sum + b_sum)
            a = np.multiply(a, avg_sum / a_sum)
            b = np.multiply(b, avg_sum / b_sum)
        else:
            raise RuntimeError("a and b vector totals do not match.")

    initial_error = _calc_error(m, a, b)
    err = 1.0
    i = 0
    while err > rel_error:
        if i > max_iterations:
            warn("Matrix balancing did not converge", RuntimeWarning)
            break
        m = _balance(m, a, 1)
        m = _balance(m, b, 0)
        err = _calc_error(m, a, b) / initial_error
        i += 1

    if data_type == "pd":
        new_df = pd.DataFrame(m, index=m_pd.index, columns=m_pd.columns)
        return new_df, err, i
    else:
        return m, err, i


def _balance(matrix: NDArray, tot: NDArray, axis: int) -> NDArray:
    """Balances a matrix using a single constraint.

    Args:
        matrix (NDArray): The matrix to be balanced
        tot (NDArray): The totals constraint
        axis (int): Direction to constrain (0 = along columns, 1 = along rows)

    Return:
        NDArray: The balanced matrix
    """
    sc = tot / (matrix.sum(axis) + EPS)
    sc = np.nan_to_num(sc)  # replace divide by 0 errors from the prev. line
    if axis:  # along rows
        matrix = np.multiply(matrix.T, sc).T
    else:  # along columns
        matrix = np.multiply(matrix, sc)
    return matrix


def _calc_error(m, a, b):
    row_sum = np.absolute(a - m.sum(1)).sum()
    col_sum = np.absolute(b - m.sum(0)).sum()
    return row_sum + col_sum


@njit(cache=False, parallel=True)
def _nbf_bucket_round(a_, decimals=0):
    a = a_.ravel()
    b = np.copy(a)

    residual = 0
    for i in prange(0, len(b)):
        b[i] = np.round(a[i] + residual, decimals)
        residual += a[i] - b[i]

    return b.reshape(a_.shape)


def matrix_bucket_rounding(
    m: Union[NDArray, pd.DataFrame],
    *,
    decimals: int = 0,
) -> Union[NDArray, pd.DataFrame]:
    """Bucket rounds to the given number of decimals.

    Args:
        m (NDArray | pandas.DataFrame): The matrix to be rounded
        decimals (int, optional): Defaults to ``0``. Number of decimal places to round to. If decimals is negative, it
            specifies the number of positions to the left of the decimal point.

    Return:
        NDArray | pandas.DataFrame: The rounded matrix
    """

    # Test if matrix is Pandas DataFrame
    data_type = ""
    m_pd = None
    if isinstance(m, pd.DataFrame):
        data_type = "pd"
        m_pd = m
        m = m_pd.values

    decimals = int(decimals)

    # I really can't think of a way to vectorize bucket rounding, so here goes the slow for loop
    b = _nbf_bucket_round(m, decimals)

    if decimals <= 0:
        b = b.astype(np.int32)

    if data_type == "pd":
        new_df = pd.DataFrame(b.reshape(m.shape), index=m_pd.index, columns=m_pd.columns)
        return new_df
    else:
        return b.reshape(m.shape)


def split_zone_in_matrix(
    base_matrix: pd.DataFrame,
    old_zone: int,
    new_zones: List[int],
    proportions: List[float],
) -> pd.DataFrame:
    """Takes a zone in a matrix (as a DataFrame) and splits it into several new zones, prorating affected cells by a
    vector of proportions (one value for each new zone). The old zone is removed.

    Args:
        base_matrix (pandas.DataFrame): The matrix to re-shape
        old_zone (int): The original zone to split
        new_zones (List[int]): The list of new zones to add
        proportions (List[float]): The proportions to split the original zone to. The list must be the same length as
            ``new_zones`` and sum to 1.0

    Returns:
        pandas.DataFrame: The re-shaped matrix
    """

    assert isinstance(base_matrix, pd.DataFrame), "Base matrix must be a DataFrame"

    old_zone = int(old_zone)
    new_zones = np.array(new_zones, dtype=np.int32)
    proportions = np.array(proportions, dtype=np.float64)

    assert len(new_zones) == len(proportions), "Proportion array must be the same length as the new zone array"
    assert len(new_zones.shape) == 1, "New zones must be a vector"
    assert base_matrix.index.equals(base_matrix.columns), "DataFrame is not a matrix"
    assert np.isclose(proportions.sum(), 1.0), "Proportions must sum to 1.0 "

    n_new_zones = len(new_zones)

    intersection_index = base_matrix.index.drop(old_zone)
    new_index = intersection_index
    for z in new_zones:
        new_index = new_index.insert(-1, z)
    new_index = pd.Index(sorted(new_index))

    new_matrix = pd.DataFrame(0, index=new_index, columns=new_index, dtype=base_matrix.dtypes.iat[0])

    # 1. Copy over the values from the regions of the matrix not being updated
    new_matrix.loc[intersection_index, intersection_index] = base_matrix

    # 2. Prorate the row corresponding to the dropped zone
    # This section (and the next) works with the underlying Numpy arrays, since they handle
    # broadcasting better than Pandas does
    original_row = base_matrix.loc[old_zone, intersection_index]
    original_row = original_row.values[:]  # Make a shallow copy to preserve shape of the original data
    original_row.shape = 1, len(intersection_index)
    proportions.shape = n_new_zones, 1
    result = pd.DataFrame(original_row * proportions, index=new_zones, columns=intersection_index)
    new_matrix.loc[result.index, result.columns] = result

    # 3. Proprate the column corresponding to the dropped zone
    original_column = base_matrix.loc[intersection_index, old_zone]
    original_column = original_column.values[:]
    original_column.shape = len(intersection_index), 1
    proportions.shape = 1, n_new_zones
    result = pd.DataFrame(original_column * proportions, index=intersection_index, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    # 4. Expand the old intrazonal
    proportions_copy = proportions[:, :]
    proportions_copy.shape = 1, n_new_zones
    proportions.shape = n_new_zones, 1

    intrzonal_matrix = proportions * proportions_copy
    intrazonal_scalar = base_matrix.at[old_zone, old_zone]

    result = pd.DataFrame(intrazonal_scalar * intrzonal_matrix, index=new_zones, columns=new_zones)
    new_matrix.loc[result.index, result.columns] = result

    return new_matrix


def aggregate_matrix(
    matrix: Union[pd.DataFrame, pd.Series],
    *,
    groups: Union[pd.Series, NDArray] = None,
    row_groups: Union[pd.Series, NDArray] = None,
    col_groups: Union[pd.Series, NDArray] = None,
    aggfunc: Callable[[Iterable[Union[int, float]]], Union[int, float]] = np.sum,
) -> Union[pd.DataFrame, pd.Series]:
    """Aggregates a matrix based on mappings provided for each axis, using a specified aggregation function.

    Args:
        matrix (pandas.DataFrame | pandas.Series): Matrix data to aggregate. DataFrames and Series with 2-level
            indices are supported
        groups (pandas.Series | NDArray, optional): Syntactic sugar to specify both row_groups and
            col_groups to use the same grouping series.
        row_groups (pandas.Series | NDArray, optional): Groups for the rows. If aggregating a DataFrame,
            this must match the index of the matrix. For a "tall" matrix, this series can match either the "full" index
            of the series, or it can match the first level of the matrix (it would be the same as if aggregating a
            DataFrame). Alternatively, an array can be provided, but it must be the same length as the DataFrame's
            index, or the full length of the Series.
        col_groups (pandas.Series | NDArray, optional): Groups for the columns. If aggregating a DataFrame,
            this must match the columns of the matrix. For a "tall" matrix, this series can match either the "full"
            index of the series, or it can match the second level of the matrix (it would be the same as if aggregating
            a DataFrame). Alternatively, an array can be provided, but it must be the same length as the DataFrame's
            columns, or the full length of the Series.
        aggfunc: The aggregation function to use. Default is np.sum.

    Returns:
        pandas.Series or pandas.DataFrame:
            The aggregated matrix, in the same type as was provided, e.g. Series -> Series, DataFrame -> DataFrame.

    Example:

        matrix:

        +-------+---+---+---+---+---+---+---+
        |       | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
        +=======+===+===+===+===+===+===+===+
        | **1** | 2 | 1 | 9 | 6 | 7 | 8 | 5 |
        +-------+---+---+---+---+---+---+---+
        | **2** | 4 | 1 | 1 | 4 | 8 | 7 | 6 |
        +-------+---+---+---+---+---+---+---+
        | **3** | 5 | 8 | 5 | 3 | 5 | 9 | 4 |
        +-------+---+---+---+---+---+---+---+
        | **4** | 1 | 1 | 2 | 9 | 4 | 9 | 9 |
        +-------+---+---+---+---+---+---+---+
        | **5** | 6 | 3 | 4 | 6 | 9 | 9 | 3 |
        +-------+---+---+---+---+---+---+---+
        | **6** | 7 | 2 | 5 | 8 | 2 | 5 | 9 |
        +-------+---+---+---+---+---+---+---+
        | **7** | 3 | 1 | 8 | 6 | 3 | 5 | 6 |
        +-------+---+---+---+---+---+---+---+

        groups:

        +-------+---+
        | **1** | A |
        +-------+---+
        | **2** | B |
        +-------+---+
        | **3** | A |
        +-------+---+
        | **4** | A |
        +-------+---+
        | **5** | C |
        +-------+---+
        | **6** | C |
        +-------+---+
        | **7** | B |
        +-------+---+

        ``new_matrix = aggregate_matrix(matrix, groups=groups)``

        new_matrix:

        +-------+----+----+----+
        |       | A  | B  | C  |
        +=======+====+====+====+
        | **A** | 42 | 28 | 42 |
        +-------+----+----+----+
        | **B** | 26 | 14 | 23 |
        +-------+----+----+----+
        | **C** | 36 | 17 | 25 |
        +-------+----+----+----+

    """
    if groups is not None:
        row_groups = groups
        col_groups = groups

    assert row_groups is not None, "Row groups must be specified"
    assert col_groups is not None, "Column groups must be specified"

    if isinstance(matrix, pd.DataFrame):
        row_groups = _prep_square_index(matrix.index, row_groups)
        col_groups = _prep_square_index(matrix.columns, col_groups)

        return _aggregate_frame(matrix, row_groups, col_groups, aggfunc)
    elif isinstance(matrix, pd.Series):
        assert matrix.index.nlevels == 2

        row_groups, col_groups = _prep_tall_index(matrix.index, row_groups, col_groups)
        return _aggregate_series(matrix, row_groups, col_groups, aggfunc)
    else:
        raise NotImplementedError()


def _prep_tall_index(target_index, row_aggregator, col_aggregator):

    if isinstance(row_aggregator, pd.Series):
        if row_aggregator.index.equals(target_index):
            row_aggregator = row_aggregator.values
        else:
            assert target_index.levels[0].equals(row_aggregator.index)
            reindexed = row_aggregator.reindex(target_index, level=0)
            row_aggregator = reindexed.values
    else:
        assert len(row_aggregator) == len(target_index)
        row_aggregator = np.array(row_aggregator)

    if isinstance(col_aggregator, pd.Series):
        if col_aggregator.index.equals(target_index):
            col_aggregator = col_aggregator.values
        else:
            assert target_index.levels[1].equals(col_aggregator.index)
            reindexed = col_aggregator.reindex(target_index, level=1)
            col_aggregator = reindexed.values
    else:
        assert len(col_aggregator) == len(target_index)
        col_aggregator = np.array(col_aggregator)

    return row_aggregator, col_aggregator


def _prep_square_index(index, aggregator):
    if isinstance(aggregator, pd.Series):
        assert aggregator.index.equals(index)
        return aggregator.values
    else:
        assert len(aggregator) == len(index)
        return np.array(aggregator)


def _aggregate_frame(matrix: pd.DataFrame, row_aggregator, col_aggregator, aggfunc):
    return matrix.groupby(row_aggregator, axis=0).aggregate(aggfunc).groupby(col_aggregator, axis=1).aggregate(aggfunc)


def _aggregate_series(matrix: pd.Series, row_aggregator, col_aggregator, aggfunc):
    return matrix.groupby([row_aggregator, col_aggregator]).aggregate(aggfunc)


def fast_stack(
    frame: pd.DataFrame,
    multi_index: pd.MultiIndex,
    *,
    deep_copy: bool = True,
) -> pd.Series:
    """Performs the same action as ``DataFrame.stack()``, but provides better performance when the target stacked index
    is known beforehand. Useful in converting a lot of matrices from "wide" to "tall" format. The inverse of
    ``fast_unstack()``.

    Notes:
        This function does not check that the entries in the multi_index are compatible with the index and columns of
        the source DataFrame, only that the lengths are compatible. It can therefore be used to assign a whole new set
        of labels to the result.

    Args:
        frame (pandas.DataFrame): The DataFrame to stack.
        multi_index (pandas.MultiIndex): The 2-level MultiIndex known ahead-of-time.
        deep_copy (bool, optional): Defaults to ``True``. A flag indicating if the returned Series should be a view of
            the underlying data (deep_copy=False) or a copy of it (deep_copy=True). A deep copy takes a little longer to
            convert and takes up more memory but preserves the original data of the DataFrame. The default value of True
            is recommended for most uses.

    Returns:
        pandas.Series: The stacked data.
    """

    assert multi_index.nlevels == 2, "Target index must be a MultiIndex with exactly 2 levels"
    assert len(multi_index) == len(frame.index) * len(frame.columns), (
        "Target index and source index and columns do " "not have compatible lengths"
    )

    array = np.ascontiguousarray(frame.values)
    array = array.copy() if deep_copy else array[:, :]
    array.shape = len(frame.index) * len(frame.columns)

    return pd.Series(array, index=multi_index)


def fast_unstack(
    series: pd.Series,
    index: pd.Index,
    columns: pd.Index,
    *,
    deep_copy: bool = True,
) -> pd.DataFrame:
    """Performs the same action as ``DataFrame.unstack()``, but provides better performance when the target unstacked
    index and columns are known beforehand. Useful in converting a lot of matrices from "tall" to "wide" format. The
    inverse of ``fast_stack()``.

    Notes:
        This function does not check that the entries in index and columns are compatible with the MultiIndex of the
        source Series, only that the lengths are compatible. It can therefore be used to assign a whole new set of
        labels to the result.

    Args:
        series (pandas.Series): The Series with 2-level MultiIndex to unstack
        index (pandas.Index): The row index known ahead-of-time
        columns (pandas.Index): The columns index known ahead-of-time.
        deep_copy (bool): Defaults to ``True``. A flag indicating if the returned DataFrame should be a view of the
            underlying data (deep_copy=False) or a copy of it (deep_copy=True). A deep copy takes a little longer to
            convert and takes up more memory but preserves the original data of the Series. The default value of True is
            recommended for most uses.

    Returns:
        pandas.DataFrame: The unstacked dat
    """

    assert series.index.nlevels == 2, "Source Series must have an index with exactly 2 levels"
    assert len(series) == len(index) * len(columns), (
        "Source index and target index and columns do not have " "compatible lengths"
    )

    array = series.values.copy() if deep_copy else series.values[:]
    array.shape = len(index), len(columns)

    return pd.DataFrame(array, index=index, columns=columns)


def _check_disaggregation_input(mapping: pd.Series, proportions: pd.Series) -> NDArray:
    assert mapping is not None
    assert proportions is not None
    assert mapping.index.equals(proportions.index)

    # Force proportions to sum to 1 by dividing by the total in each parent
    parent_totals = (
        proportions.groupby(mapping)  # Group the proportions by parent zones
        .sum()  # Sum the total for each parent
        .reindex(mapping)  # Reindex for all child zones
        .values  # Get the ndarray to avoid index alignment problems
    )

    return proportions.values / parent_totals


def disaggregate_matrix(
    matrix: pd.DataFrame,
    *,
    mapping: pd.Series = None,
    proportions: pd.Series = None,
    row_mapping: pd.Series = None,
    row_proportions: pd.Series = None,
    col_mapping: pd.Series = None,
    col_proportions: pd.Series = None,
) -> pd.DataFrame:
    """Split multiple rows and columns in a matrix all at once. The cells in the matrix MUST be numeric, but the row
    and column labels do not.

    Args:
        matrix (pandas.DataFrame): The input matrix to disaggregate
        mapping (pandas.Series, optional): Dict-like Series of "New label" : "Old label". Sets both the row_mapping and
            col_mapping variables if provided (resulting in a square matrix).
        proportions (pandas.Series, optional): Dict-like Series of "New label": "Proportion of old label". Its index
            must match the index of the mapping argument. Sets both the row_proportions and col_proportions arguments
            if provided.
        row_mapping (pandas.Series, optional): Same as mapping, except applied only to the rows.
        row_proportions (pandas.Series, optional): Same as proportions, except applied only to the rows
        col_mapping (pandas.Series, optional): Same as mapping, except applied only to the columns.
        col_proportions (pandas.Series, optional): Same as proportions, except applied only to the columns

    Returns:
        pandas.DataFrame: An expanded DataFrame with the new indices. The new matrix will sum to the same total as the original.

    Examples:

        df:

        +---+----+----+----+
        |   | A  | B  | C  |
        +===+====+====+====+
        | A | 10 | 30 | 20 |
        +---+----+----+----+
        | B | 20 | 10 | 10 |
        +---+----+----+----+
        | C | 30 | 20 | 20 |
        +---+----+----+----+

        correspondence:

        +-----+-----+------+
        | new | old | prop |
        +=====+=====+======+
        | A1  |  A  | 0.25 |
        +-----+-----+------+
        | A2  |  A  | 0.75 |
        +-----+-----+------+
        | B1  |  B  | 0.55 |
        +-----+-----+------+
        | B2  |  B  | 0.45 |
        +-----+-----+------+
        | C1  |  C  | 0.62 |
        +-----+-----+------+
        | C2  |  C  | 0.38 |
        +-----+-----+------+

        ``new_matrix = disaggregate_matrix(df, mapping=correspondence['old'], proportions=correspondence['prop'])``

        new_matrix:

        +-----+-------+-------+--------+--------+-------+-------+
        | new |  A1   | A2    | B1     | B2     | C1    | C2    |
        +=====+=======+=======+========+========+=======+=======+
        |  A1 | 0.625 | 1.875 |  4.125 |  3.375 | 3.100 | 1.900 |
        +-----+-------+-------+--------+--------+-------+-------+
        |  A2 | 1.875 | 5.625 | 12.375 | 10.125 | 9.300 | 5.700 |
        +-----+-------+-------+--------+--------+-------+-------+
        |  B1 | 2.750 | 8.250 |  3.025 |  2.475 | 3.410 | 2.090 |
        +-----+-------+-------+--------+--------+-------+-------+
        |  B2 | 2.250 | 6.750 |  2.475 |  2.025 | 2.790 | 1.710 |
        +-----+-------+-------+--------+--------+-------+-------+
        |  C1 | 4.650 | 13.95 |  6.820 |  5.580 | 7.688 | 4.712 |
        +-----+-------+-------+--------+--------+-------+-------+
        |  C2 | 2.850 |  8.55 |  4.180 |  3.420 | 4.712 | 2.888 |
        +-----+-------+-------+--------+--------+-------+-------+

    """

    # Check that all inputs are specified
    if mapping is not None:
        row_mapping, col_mapping = mapping, mapping
    if proportions is not None:
        row_proportions, col_proportions = proportions, proportions

    row_proportions = _check_disaggregation_input(row_mapping, row_proportions)
    col_proportions = _check_disaggregation_input(col_mapping, col_proportions)

    # Validate inputs
    new_rows = row_mapping.index
    new_cols = col_mapping.index

    # Get raw indexers for NumPy & lookup the value in each parent cell
    row_indexer = matrix.index.get_indexer(row_mapping)[:, np.newaxis]
    col_indexer = matrix.columns.get_indexer(col_mapping)[np.newaxis, :]
    parent_cells = matrix.values[row_indexer, col_indexer]

    # Convert proportions to 2D vectors
    row_proportions = row_proportions[:, np.newaxis]
    col_proportions = col_proportions[np.newaxis, :]

    # Multiply each parent cell by its disaggregation proportion & return
    result_matrix = ne.evaluate("parent_cells * row_proportions * col_proportions")

    result_matrix = pd.DataFrame(result_matrix, index=new_rows, columns=new_cols)
    return result_matrix
