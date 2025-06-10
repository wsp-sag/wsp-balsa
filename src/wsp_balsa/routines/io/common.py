from __future__ import annotations

from contextlib import contextmanager
from io import FileIO
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def coerce_matrix(
    matrix: Union[NDArray, pd.DataFrame, pd.Series],
    *,
    allow_raw: bool = True,
    force_square: bool = True,
) -> NDArray:
    """Infers a NumPy array from given input

    Args:
        matrix (NDArray | pandas.DataFrame | pandas.Series): Matrix data to coerce its shape into a 2D ndarray
        allow_raw (bool, optional): Defaults to ``True``.
        force_square (bool, optional): Defaults to ``True``. A flag to check if the resulting matrix is square in shape

    Returns:
        NDArray: A 2D ndarray of type float32
    """
    if isinstance(matrix, pd.DataFrame):
        if force_square:
            if not matrix.index.equals(matrix.columns):
                raise ValueError("Rows and columns in `matrix` are not identical")
        return matrix.values.astype(np.float32)
    elif isinstance(matrix, pd.Series):
        if matrix.index.nlevels != 2:
            raise ValueError("Cannot infer a matrix from a Series with more or fewer than 2 levels")
        wide = matrix.unstack()

        union = wide.index | wide.columns
        wide = wide.reindex_axis(union, fill_value=0.0, axis=0).reindex_axis(union, fill_value=0.0, axis=1)
        return wide.values.astype(np.float32)

    if not allow_raw:
        raise NotImplementedError()

    matrix = np.array(matrix, dtype=np.float32)
    if len(matrix.shape) != 2:
        raise ValueError("`matrix` is not 2-dimensional")
    i, j = matrix.shape
    if i != j:
        raise ValueError("`matrix` is not square")

    return matrix


def expand_array(
    a: NDArray,
    n: NDArray,
    *,
    axis: int = None,
) -> NDArray:
    """Expands an array across all dimensions by a set amount

    Args:
        a (NDArray): The array to expand
        n (NDArray): The (non-negative) number of items to expand by.
        axis (int, optional): Defaults to ``None``. The axis to expand along, or None to expand along all axes.

    Returns:
        NDArray: The expanded array
    """

    if axis is None:
        new_shape = [dim + n for dim in a.shape]
    else:
        new_shape = []
        for i, dim in enumerate(a.shape):
            dim += n if i == axis else 0
            new_shape.append(dim)

    out = np.zeros(new_shape, dtype=a.dtype)

    indexer = [slice(0, dim) for dim in a.shape]
    out[indexer] = a

    return out


@contextmanager
def open_file(file_handle: Union[str, Path, FileIO], **kwargs):
    """Context manager for opening files provided as several different types. Supports a file handler as a str, unicode,
    ``pathlib.Path``, or an already-opened handler.

    Args:
        file_handle (str | Path | FileIO): The item to be opened or is already open.
        **kwargs: Keyword args passed to ``open()``. Usually mode='w'.

    Yields:
        File:
            The opened file handler. Automatically closed once out of context.
    """
    opened = False
    if isinstance(file_handle, str):
        f = open(file_handle, **kwargs)
        opened = True
    elif Path is not None and isinstance(file_handle, Path):
        f = file_handle.open(**kwargs)
        opened = True
    else:
        f = file_handle

    try:
        yield f
    finally:
        if opened:
            f.close()
