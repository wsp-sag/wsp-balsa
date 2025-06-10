from __future__ import annotations

from multiprocessing import cpu_count
from threading import Thread
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):

        def decorator(func):
            return func

        return decorator


_NULL_INDEX = -1
_NEG_INF = -np.inf


def _get_breaks(n_items: int, n_workers: int) -> List[Tuple[int, int]]:
    """Gets starts and stops for breaks similar to numpy.array_slice()"""
    div, remainder = divmod(n_items, n_workers)
    slices = []
    bottom = 0
    for i in range(n_workers):
        top = bottom + div
        top += int(i < remainder)
        slices.append((bottom, top))
        bottom = top
    return slices


@njit(cache=True)
def _update_heap(utilities: NDArray, zones: NDArray, new_u: float, new_zone: int):
    """Inserts a value into a sorted array, maintaining sort order, and the number of items in that array. The array
    is sorted lowest-to-highest"""
    i = 0
    top = len(utilities)
    while i < top:
        current_u = utilities[i]
        if new_u < current_u:
            break
        i += 1
    if i <= 0:
        return
    for j in range(i - 1):
        utilities[j] = utilities[j + 1]
        zones[j] = zones[j + 1]
    utilities[i - 1] = new_u
    zones[i - 1] = new_zone


@njit(nogil=True, cache=True)
def _nbf_twopart_worker(
    pk_costs: NDArray,
    kq_costs: NDArray,
    add_pkq_costs: NDArray,
    result_costs: NDArray,
    result_indices: NDArray,
    flag_array: NDArray,
    start: int,
    stop: int,
    n: int,
):
    """Performance-tuned Numba function to operate on its own thread"""
    n_origins, n_intermediate = pk_costs.shape
    n_destinations = kq_costs.shape[1]

    # Allocate the sorted heap of costs (and associated indices) once per thread
    cost_heap = np.full(shape=n, fill_value=_NEG_INF)
    zones_heap = np.full(shape=n, fill_value=_NULL_INDEX)

    for offset in range(start, stop):
        if not flag_array[offset]:
            continue  # Skip certain ODs
        p, q = divmod(offset, n_destinations)

        # Reset the heap for this OD
        cost_heap[:] = _NEG_INF
        zones_heap[:] = _NULL_INDEX

        for k in range(n_intermediate):
            if k >= n_intermediate:
                print("ERR", offset, p, q, k)
                raise AssertionError()

            interim_cost = pk_costs[p, k] + kq_costs[k, q]
            if add_pkq_costs is not None:
                interim_cost = interim_cost + add_pkq_costs[p, k, q]

            # In general, for problems where (n_origins * n_destinations) >> k, most values will not be in the top
            # k. So quickly check against the lowest cost in the heap to avoid calling the updater func
            if (interim_cost < cost_heap[0]) or (interim_cost == _NEG_INF):
                continue
            _update_heap(cost_heap, zones_heap, interim_cost, k)

        result_costs[p, q, :] = cost_heap
        result_indices[p, q, :] = zones_heap


@njit(nogil=True, cache=True)
def _nbf_twopart_subset_worker(
    pk_costs: NDArray,
    kq_costs: NDArray,
    add_pkq_costs: NDArray,
    subset_pk_costs: NDArray,
    result_costs: NDArray,
    result_indices: NDArray,
    flag_array: NDArray,
    start: int,
    stop: int,
    n_subset: int,
    n_final: int,
):
    """Performance-tuned Numba function to operate on its own thread"""
    n_origins, n_intermediate = pk_costs.shape
    n_destinations = kq_costs.shape[1]

    # Allocate the sorted heap of utilities (and associated indices) once per thread
    subset_cost_heap = np.full(shape=n_subset, fill_value=_NEG_INF)
    subset_zones_heap = np.full(shape=n_subset, fill_value=_NULL_INDEX)

    cost_heap = np.full(shape=n_final, fill_value=_NEG_INF)
    zone_heap = np.full(shape=n_final, fill_value=_NULL_INDEX)

    for offset in range(start, stop):
        if not flag_array[offset]:
            continue  # Skip certain ODs
        p, q = divmod(offset, n_destinations)

        # Reset the heap for this OD
        subset_cost_heap[:] = _NEG_INF
        subset_zones_heap[:] = _NULL_INDEX

        cost_heap[:] = _NEG_INF
        zone_heap[:] = _NULL_INDEX

        # Find the superset of zones by subset cost
        for k in range(n_intermediate):
            if k >= n_intermediate:
                print("ERR", offset, p, q, k)
                raise AssertionError()

            interim_pk_cost = subset_pk_costs[p, k]
            interim_cost = kq_costs[k, q]
            # This zone is not available if value is greater than zero (Given cost value should be negative now)
            if interim_pk_cost >= 0:
                continue

            # In general, for problems where (n_origins * n_destinations) >> k, most values will not be in the top
            # k. So quickly check against the lowest utility in the heap to avoid calling the updater func
            # Interim utility needs to be checked to ensure the path is a viable choice
            if (interim_pk_cost < subset_cost_heap[0]) or (interim_cost == _NEG_INF):
                continue
            _update_heap(subset_cost_heap, subset_zones_heap, interim_pk_cost, k)

        # Compute the composite utility for the subset zones
        for k in subset_zones_heap:
            interim_cost = pk_costs[p, k] + kq_costs[k, q]
            if add_pkq_costs is not None:
                interim_cost = interim_cost + add_pkq_costs[p, k, q]

            # In general, for problems where (n_origins * n_destinations) >> k, most values will not be in the top
            # k. So quickly check against the lowest utility in the heap to avoid calling the updater func
            if (interim_cost < cost_heap[0]) or (interim_cost == _NEG_INF):
                continue
            _update_heap(cost_heap, zone_heap, interim_cost, k)

        result_costs[p, q, :] = cost_heap
        result_indices[p, q, :] = zone_heap


def _validate_pk_kq_tables(pk_table: pd.DataFrame, kq_table: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:

    if pk_table.index.nlevels != 2:
        raise RuntimeError("pk table index must have two levels")
    if kq_table.index.nlevels != 2:
        raise RuntimeError("kq table index must have two levels")

    # Take the unique index of each level, as Pandas can return more items than is present, if the frame is a slice
    origin_zones = pk_table.index.unique(level=0)
    intermediate_zones = pk_table.index.unique(level=1)
    destination_zones = kq_table.index.unique(level=1)

    # Check that the access and egress tables have compatible indices
    if not intermediate_zones.equals(kq_table.index.unique(level=0)):
        raise RuntimeError("pk index level 2 and kq index level 1 must be the same")

    return origin_zones, intermediate_zones, destination_zones


def best_intermediate_zones(
    pk_table: pd.DataFrame,
    kq_table: pd.DataFrame,
    cost_col: str,
    *,
    n: int = 1,
    add_pkq_cost: NDArray = None,
    flag_array: NDArray = None,
    maximize: bool = True,
    null_index: int = 0,
    other_columns: bool = True,
    intermediate_name: str = "intermediate_zone",
    availability_name: str = "available",
    n_threads: int = 1,
    squeeze: bool = True,
) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Numba-accelerated.

    Triple-index operation for two matrices, finding the most- or least-cost intermediate zones. Takes a first leg
    matrix ("pk") and a second leg matrix ("kq") to produce a combined "pq" matrix with the best intermediate "k".
    Also works to construct multiple "pq" matrices for the top _n_ intermediate "k" zones.

    There is no restriction on the label dtypes, as long as the "pk" (leg 1) and "kq" (leg 2) tables share the same
    "k" index.

    Both the input matrices must be provided in "tall" format - as Pandas Series with a 2-level MultiIndex.
    Essentially, the "pk" (leg 1) and "kq" (leg 2) tables are DataFrames with multiple matrices defined within. The
    output table(s) are also returned in a tall format.

    When constructing the result tables, columns in the "pk" (leg 1) and "kq" (leg 2) tables are "carried forward"
    such that the results columns will be the union of columns in the input tables. Columns in one table only will
    be carried forward unmodified and retain their data type. Columns in both tables will be added together, and
    thus MUST be numeric.

    In the specified cost column, a value of `-inf` (or `inf` when minimizing) is respected as the sentinel value
    for unavailable. "pk" or "kq" interchanges with this sentinel value will not be considered.

    Args:
        pk_table (pd.DataFrame): A DataFrame with 2-level MultiIndex of the shape ((p, k), A). Must include the
            specified cost column
        kq_table (pd.DataFrame): A DataFrame with 2-level MultiIndex of the shape ((k, q), E). Must include the
            specified cost column
        cost_col (str): Name of the column in the access and egress table to use as the cost to
            minimize/maximize. Values of `+/- inf` are respected to indicate unavailable choices.
        n (int, optional): Defaults to ``1``. The number of ranks to return (e.g., find the _n_ best intermediate
            zones). If n <= 0, it will be corrected to 1.
        add_pkq_cost (ndarray): A 3-dimensional numpy array containing additional two-part path ("pkq") costs to be
            included in triple-index operation.
        flag_array (ndarray, optional): Defaults to ``None``. An array of boolean flags indicating the "pq"
            zone pairs to evaluate.
        maximize (bool, optional): Defaults to ``True``. If True, this function maximize the result. If False, it
            minimizes it.
        null_index (int, optional): Defaults to ``0``. Fill value used if NO intermediate zone is available.
        other_columns (bool, optional): Defaults to ``True``. If True, the result DataFrame will include all columns
            in the "pk" and "kq" tables. The result table will be of the shape ((p, q), A | E + 3)
        intermediate_name (str, optional): Defaults to ``'intermediate_zone'``. Name of the column in the result
            table containing the selected intermediate zone.
        availability_name (str, optional): Defaults to ``'available'``. Name of the column in the result table
            containing a flag whether ANY intermediate zone was found to be available.
        n_threads (int, optional): Defaults to ``1``. Number of threads to use.
        squeeze (bool, optional): Defaults to ``True``. If ``n == 1`` and ``squeeze=True``, a single DataFrame is
            returned. Otherwise, a dictionary of DataFrames will be returned.

    Returns:
        DataFrame: If n == 1 and squeeze=True. A DataFrame of the shape ((p, q), A | E + 3), containing the
            intermediate "k" zone selected, the associated max/min cost, and a flag indicating its availability.
            Additional columns from the "pk" (leg 1) and "kq" (leg 2) tables, indexed for the appropriately chosen
            intermediate zone, will also be included if other_columns=True.
        Dict[int, DataFrame]: If n > 1. The keys represent the ranks, so result[1] is the best intermediate zone,
            result[2] is the second-best, etc. The value DataFrames are in the same format as if n == 1, just with
            different intermediate zones chosen.
    """

    # Check inputs
    n = max(1, n)
    n_threads = int(np.clip(n_threads, 1, cpu_count()))

    origin_zones, intermediate_zones, destination_zones = _validate_pk_kq_tables(pk_table, kq_table)

    # Set up the raw data
    n_origins, n_intermediate, n_destinations = len(origin_zones), len(intermediate_zones), len(destination_zones)
    pk_cost = pk_table[cost_col].to_numpy().reshape([n_origins, n_intermediate]).astype(np.float64)
    kq_cost = kq_table[cost_col].to_numpy().reshape([n_intermediate, n_destinations]).astype(np.float64)
    if not maximize:  # Invert the cost to use code set to maximize
        pk_cost *= -1
        kq_cost *= -1

    result_cost = np.zeros(shape=(n_origins, n_destinations, n), dtype=np.float64)
    result_indices = np.zeros(shape=(n_origins, n_destinations, n), dtype=np.int16)

    if add_pkq_cost is not None:
        if add_pkq_cost.shape != (n_origins, n_intermediate, n_destinations):
            raise RuntimeError(
                "Shape of `add_pkq_cost` does not match the number of origin, intermediate, and "
                "destination zones found in `pk_table` and `kq_table`"
            )
        add_pkq_cost = add_pkq_cost.astype(np.float64)

    if flag_array is None:
        flag_array = np.full(n_origins * n_destinations, fill_value=True)
    else:
        if not len(flag_array) == (n_origins * n_destinations):
            raise RuntimeError("Length of `flag_array` incompatible with size of `pk_table` and `kq_table`")
        flag_array = flag_array.astype(np.bool_)

    # Run the accelerated function
    breaks = _get_breaks(n_origins * n_destinations, n_threads)
    threads = np.empty(n_threads, dtype="O")
    for i, (start, stop) in enumerate(breaks):
        t = Thread(
            target=_nbf_twopart_worker,
            args=[pk_cost, kq_cost, add_pkq_cost, result_cost, result_indices, flag_array, start, stop, n],
        )
        t.start()
        threads[i] = t
    for t in threads:
        t.join()

    return _combine_tables(
        pk_table,
        kq_table,
        result_cost,
        result_indices,
        flag_array,
        origin_zones,
        intermediate_zones,
        destination_zones,
        cost_col,
        availability_name,
        intermediate_name,
        other_columns=other_columns,
        squeeze=squeeze,
        null_index=null_index,
    )


def best_intermediate_subset_zones(
    pk_subset_cost: pd.Series,
    pk_table: pd.DataFrame,
    kq_table: pd.DataFrame,
    cost_col: str,
    *,
    n_subset: int = 1,
    n_final: int = 1,
    add_pkq_cost: NDArray = None,
    flag_array: NDArray = None,
    maximize_subset: bool = True,
    maximize_final: bool = True,
    null_index: int = 0,
    other_columns: bool = True,
    intermediate_name: str = "intermediate_zone",
    availability_name: str = "available",
    n_threads: int = 1,
    squeeze: bool = True,
) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Numba-accelerated.

    Triple-index operation for two matrices, finding the most- or least-cost intermediate zones from a subset.
    Takes a first leg matrix ("pk") and a second leg matrix ("kq") to produce a combined "pq" matrix with the best
    intermediate "k". Also works to construct multiple "pq" matrices for the top _n_final_ intermediate "k" zones.

    There is no restriction on the label dtypes, as long as the "pk" (leg 1) and "kq" (leg 2) tables share the same
    "k" index.

    Both the input matrices must be provided in "tall" format - as Pandas Series with a 2-level MultiIndex.
    Essentially, the "pk" (leg 1) and "kq" (leg 2) tables are DataFrames with multiple matrices defined within. The
    output table(s) are also returned in a tall format.

    When constructing the result tables, columns in the "pk" (leg 1) and "kq" (leg 2) tables are "carried forward"
    such that the results columns will be the union of columns in the input tables. Columns in one table only will
    be carried forward unmodified and retain their data type. Columns in both tables will be added together, and
    thus MUST be numeric.

    In the specified cost column, a value of `-inf` (or `inf` when minimizing) is respected as the sentinel value
    for unavailable. "pk" or "kq" interchanges with this sentinel value will not be considered.

    Args:
        pk_subset_cost (pd.Series): A Series with 2-level MultiIndex containing values to use for subsetting
        pk_table (pd.DataFrame): A DataFrame with 2-level MultiIndex of the shape ((p, k), A). Must include the
            specified cost column
        kq_table (pd.DataFrame): A DataFrame with 2-level MultiIndex of the shape ((k, q), E). Must include the
            specified cost column
        cost_col (str): Name of the column in the access and egress table to use as the cost to
            minimize/maximize. Values of `+/- inf` are respected to indicate unavailable choices.
        n_subset (int, optional): Defaults to ``1``. The number of intermediate ranks to subset (e.g., find the
            _n_subset_ best intermediate zones to select the _n_final_ best intermediate zones). If n_subset <= 0,
            it will be corrected to 1.
        n_final (int, optional): Defaults to ``1``. The number of ranks to return (e.g., find the _n_final_ best
            intermediate zones). If n_final <= 0, it will be corrected to 1.
        add_pkq_cost (ndarray): A 3-dimensional numpy array containing additional two-part path ("pkq") costs to be
            included in triple-index operation.
        flag_array (ndarray, optional): Defaults to ``None``. An array of boolean flags indicating the "pq"
            zone pairs to evaluate.
        maximize_subset (bool, optional): Defaults to ``True``. If True, this function maximize the result when
            determining the "pk" selection. If False, it minimizes it.
        maximize_final (bool, optional): Defaults to ``True``. If True, this function maximize the result when
            determining the "pq" selection. If False, it minimizes it.
        null_index (int, optional): Defaults to ``0``. Fill value used if NO intermediate zone is available.
        other_columns (bool, optional): Defaults to ``True``. If True, the result DataFrame will include all columns
            in the "pk" and "kq" tables. The result table will be of the shape ((p, q), A | E + 3)
        intermediate_name (str, optional): Defaults to ``'intermediate_zone'``. Name of the column in the result
            table containing the selected intermediate zone.
        availability_name (str, optional): Defaults to ``'available'``. Name of the column in the result table
            containing a flag whether ANY intermediate zone was found to be available.
        n_threads (int, optional): Defaults to ``1``. Number of threads to use.
        squeeze (bool, optional): Defaults to ``True``. If ``n_final == 1`` and ``squeeze=True``, a single DataFrame
            is returned. Otherwise, a dictionary of DataFrames will be returned.

    Returns:
        DataFrame: If n_final == 1 and squeeze=True. A DataFrame of the shape ((p, q), A | E + 3), containing the
            intermediate "k" zone selected, the associated max/min cost, and a flag indicating its availability.
            Additional columns from the "pk" (leg 1) and "kq" (leg 2) tables, indexed for the appropriately chosen
            intermediate zone, will also be included if other_columns=True.
        Dict[int, DataFrame]: If n_final > 1. The keys represent the ranks, so result[1] is the best intermediate
            zone, result[2] is the second-best, etc. The value DataFrames are in the same format as if n_final == 1,
            just with different intermediate zones chosen.
    """

    # Check inputs
    n_subset = max(1, n_subset)
    n_final = max(1, n_final)
    if n_subset < n_final:
        raise ValueError("`n_subset` must be >= `n_final`")
    n_threads = int(np.clip(n_threads, 1, cpu_count()))

    origin_zones, intermediate_zones, destination_zones = _validate_pk_kq_tables(pk_table, kq_table)
    if not pk_subset_cost.index.equals(pk_table.index):
        raise RuntimeError("`pk_subset_cost` and `pk_table` have incompatible indices")

    # Set up the raw data
    n_origins, n_intermediate, n_destinations = len(origin_zones), len(intermediate_zones), len(destination_zones)
    pk_subset_cost = pk_subset_cost.to_numpy().reshape([n_origins, n_intermediate]).astype(np.float64)
    if not maximize_subset:  # Invert the cost to use code set to maximize
        pk_subset_cost *= -1
    pk_cost = pk_table[cost_col].to_numpy().reshape([n_origins, n_intermediate]).astype(np.float64)
    kq_cost = kq_table[cost_col].to_numpy().reshape([n_intermediate, n_destinations]).astype(np.float64)
    if not maximize_final:  # Invert the cost to use code set to maximize
        pk_cost *= -1
        kq_cost *= -1

    result_cost = np.zeros(shape=(n_origins, n_destinations, n_final), dtype=np.float64)
    result_indices = np.zeros(shape=(n_origins, n_destinations, n_final), dtype=np.int16)

    if add_pkq_cost is not None:
        if add_pkq_cost.shape != (n_origins, n_intermediate, n_destinations):
            raise RuntimeError(
                "Shape of `add_pkq_cost` does not match the number of origin, intermediate, and "
                "destination zones found in `pk_table` and `kq_table`"
            )
        add_pkq_cost = add_pkq_cost.astype(np.float64)

    if flag_array is None:
        flag_array = np.full(n_origins * n_destinations, fill_value=True)
    else:
        if not len(flag_array) == (n_origins * n_destinations):
            raise RuntimeError("Length of `flag_array` incompatible with size of `pk_table` and `kq_table`")
        flag_array = flag_array.astype(np.bool_)

    # Run the accelerated function
    breaks = _get_breaks(n_origins * n_destinations, n_threads)
    threads = np.empty(n_threads, dtype="O")
    for i, (start, stop) in enumerate(breaks):
        t = Thread(
            target=_nbf_twopart_subset_worker,
            args=[
                pk_cost,
                kq_cost,
                add_pkq_cost,
                pk_subset_cost,
                result_cost,
                result_indices,
                flag_array,
                start,
                stop,
                n_subset,
                n_final,
            ],
        )
        t.start()
        threads[i] = t
    for t in threads:
        t.join()

    return _combine_tables(
        pk_table,
        kq_table,
        result_cost,
        result_indices,
        flag_array,
        origin_zones,
        intermediate_zones,
        destination_zones,
        cost_col,
        availability_name,
        intermediate_name,
        other_columns=other_columns,
        squeeze=squeeze,
        null_index=null_index,
    )


def _reshape_series(
    series: pd.Series, n_rows: int, n_cols: int, row_indexer: NDArray, col_indexer: NDArray
) -> Union[NDArray, pd.Categorical]:
    if isinstance(series, pd.CategoricalDtype):
        codes = series.cat.codes
        reshaped = codes.to_numpy().reshape([n_rows, n_cols])
        flat = reshaped[row_indexer, col_indexer]
        return pd.Categorical.from_codes(flat, categories=series.cat.categories)
    else:
        reshaped = series.to_numpy().reshape([n_rows, n_cols])
        return reshaped[row_indexer, col_indexer]


def _combine_tables(
    pk_table: pd.DataFrame,
    kq_table: pd.DataFrame,
    result_cost: NDArray,
    result_indices: NDArray,
    flag_array: NDArray,
    origin_zones: pd.Index,
    intermediate_zones: pd.Index,
    destination_zones: pd.Index,
    cost_col: str,
    avail_col: str,
    intermediate_name: str,
    *,
    other_columns: bool = True,
    squeeze: bool = True,
    null_index: int = 0,
) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:

    n_origins, n_intermediate, n_destinations = len(origin_zones), len(intermediate_zones), len(destination_zones)
    n_selected: int = result_indices.shape[2]

    # Construct composite result tables
    if other_columns:
        remaining_columns = set(pk_table.columns.union(kq_table.columns)) - {cost_col, intermediate_name, avail_col}
    else:
        remaining_columns = set()

    # Collect columns for the number of selected zones
    row_index = pd.MultiIndex.from_product([origin_zones, destination_zones])
    pk_indexer = np.repeat(np.arange(n_origins), n_destinations)
    kq_indexer = np.tile(np.arange(n_destinations), n_origins)
    if flag_array is not None:
        row_index = row_index[flag_array]
        pk_indexer = pk_indexer[flag_array]
        kq_indexer = kq_indexer[flag_array]

    tables = {}
    for i in range(n_selected):
        table = pd.DataFrame(index=row_index)

        offsets_i = result_indices[:, :, i]  # 2D indexer for this (i ∈ k) path
        flat_offsets_i = offsets_i.flatten()  # Convert to 1D indexer
        raw_cost = result_cost[:, :, i].flatten()
        if flag_array is not None:
            flat_offsets_i = flat_offsets_i[flag_array]
            raw_cost = raw_cost[flag_array]

        availability_i = flat_offsets_i != _NULL_INDEX
        intermediate_result_i = intermediate_zones.to_numpy().take(flat_offsets_i)
        intermediate_result_i[~availability_i] = null_index

        table[intermediate_name] = intermediate_result_i
        table[avail_col] = availability_i
        table[cost_col] = raw_cost

        # Loop through any additional columns, adding together if they exist in both tables
        for column in sorted(remaining_columns):
            in_pk_table = column in pk_table
            in_kq_table = column in kq_table

            pk_component = 0
            if in_pk_table:
                pk_component = _reshape_series(pk_table[column], n_origins, n_intermediate, pk_indexer, flat_offsets_i)

            kq_component = 0
            if in_kq_table:
                kq_component = _reshape_series(
                    kq_table[column], n_intermediate, n_destinations, flat_offsets_i, kq_indexer
                )

            if in_pk_table and in_kq_table:
                composite_data = pk_component + kq_component
            elif in_pk_table:
                composite_data = pk_component
            elif in_kq_table:
                composite_data = kq_component
            else:
                raise RuntimeError("This shouldn't happen")
            table[column] = composite_data

        tables[n_selected - i] = table

    if (n_selected == 1) and squeeze:
        return tables[1]
    return tables
