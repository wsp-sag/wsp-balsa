from __future__ import annotations

import re
from collections.abc import Sequence
from keyword import kwlist
from typing import Dict, List, Union

from pandas import DataFrame, Index, MultiIndex, Series


def reindex_series(
    series: Series,
    target_series: Series,
    *,
    source_levels: List[int] = None,
    target_levels: List[int] = None,
    fill_value: Union[int, float] = None,
) -> Series:
    # Make shallow copies of the source and target series in case their indexes need to be changed
    series = series.copy(deep=False)
    target_series = target_series.copy(deep=False)

    if series.index.nlevels > 1 and source_levels is not None:
        arrays = [series.index.get_level_values(level) for level in source_levels]
        series.index = MultiIndex.from_arrays(arrays)

    if target_series.index.nlevels > 1 and target_levels is not None:
        arrays = [target_series.index.get_level_values(level) for level in target_levels]
        target_series.index = MultiIndex.from_arrays(arrays)

    reindexed = series.reindex(target_series.values, fill_value=fill_value)
    reindexed.index = target_series.index

    return reindexed


def align_categories(
    items: Sequence[Union[Series, DataFrame]],
):
    """Pre-processing step for ``pd.concat()`` which attempts to align any Categorical series in the sequence to using
    the same set of categories. It passes through the sequence twice: once to accumulate the complete set of all
    categories used in the sequence; and a second time to modify the sequence's contents to use this full set. The
    contents of the sequence are modified in-place.

    Note:
        The resulting categories will be lex-sorted (based on the ``sorted()`` builtin)

    Args:
        items (Sequence[pandas.Series | pandas.DataFrame]): Any sequence of Series or DataFrame objects (anything that
            is acceptable to ``pandas.concat()``)
    """
    item_type = None
    for item in items:
        if item_type is None:
            if isinstance(item, DataFrame):
                item_type = DataFrame
            elif isinstance(item, Series):
                item_type = Series
            else:
                raise TypeError(type(item))
        else:
            assert isinstance(item, item_type)

    if item_type is Series:
        _align_series_categories(items)
    else:
        column_categories = _enumerate_frame_categories(items)
        _align_frame_categories(items, column_categories)


def _align_series_categories(series_list: Sequence[Series]):
    all_categories = set()
    for series in series_list:
        if not hasattr(series, "cat"):
            raise TypeError()
        all_categories |= set(series.cat.categories)

    sorted_categories = sorted(all_categories)
    for series in series_list:
        missing_categories = all_categories.difference(series.cat.categories)
        if missing_categories:
            series.cat.add_categories(missing_categories, inplace=True)
        series.cat.reorder_categories(sorted_categories, inplace=True)


def _enumerate_frame_categories(frames: Sequence[DataFrame]) -> Dict[str, set]:
    column_categories = {}
    for frame in frames:
        for col_name, series in frame.items():
            if not hasattr(series, "cat"):
                continue
            categories = set(series.cat.categories)

            if col_name not in column_categories:
                column_categories[col_name] = categories
            else:
                column_categories[col_name] |= categories
    return column_categories


def _align_frame_categories(frames: Sequence[DataFrame], column_categories: Dict[str, set]):
    for col_name, all_categories in column_categories.items():
        sorted_categories = sorted(all_categories)
        for frame in frames:
            if col_name not in frame:
                continue
            s = frame[col_name]
            missing_categories = all_categories.difference(s.cat.categories)
            if missing_categories:
                s.cat.add_categories(missing_categories, inplace=True)
            s.cat.reorder_categories(sorted_categories, inplace=True)


def sum_df_sequence(
    items: Sequence[DataFrame],
    *,
    fill_value: Union[int, float] = 0,
) -> DataFrame:
    """Sums over a sequence of DataFrames, even if they have different indexes or columns, filling in 0 (or a value of
    your choice) for missing rows or columns. Useful when you have a sequence of DataFrames which are supposed to have
    the same indexes and columns but might be missing a few values.

    Args:
        items (Sequence[pandas.DataFrame]): Any sequence of DataFrames, ordered or unordered.
        fill_value (int | float, optional): Defaults to ``0``. The value to use for missing cells.

    Returns:
        pandas.DataFrame: The sum over all items in seq.
    """
    common_index = Index([])
    common_columns = Index([])
    accumulator = DataFrame()

    for df in items:
        if not df.index.equals(common_index):
            common_index |= df.index
            accumulator = accumulator.reindex(common_index, axis=0, fill_value=fill_value)
            df = df.reindex(common_index, axis=0, fill_value=fill_value)
        if not df.columns.equals(common_columns):
            common_columns |= df.columns
            accumulator = accumulator.reindex(common_columns, axis=1, fill_value=fill_value)
            df = df.reindex(common_columns, axis=1, fill_value=fill_value)
        accumulator += df
    return accumulator


def is_identifier(name: str) -> bool:
    """Tests that the name is a valid Python variable name and does not collide with reserved keywords

    Args:
        name (str): Name to test

    Returns:
        bool: If the name is 'Pythonic'
    """

    return name.isidentifier() and name not in kwlist


def _tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def _alphanum_key(s):
    """Turn a string into a list of string and number chunks (eg. "z23a" -> ["z", 23, "a"])"""
    return [_tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(
    l: List[str],
) -> List[str]:
    """Sort the given list of strings in the way that humans expect.

    Args:
        l (List[str]): List of strings to sort.

    Returns:
        List[str]: The sorted list of strings
    """
    l = l.copy()
    l.sort(key=_alphanum_key)
    return l
