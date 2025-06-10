from __future__ import annotations

from io import FileIO
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .common import coerce_matrix, expand_array, open_file


def _infer_fortran_zones(n_words):
    """Returns the inverse of n_words = matrix_size * (matrix_size + 1)"""
    n = int(0.5 + ((1 + 4 * n_words) ** 0.5) / 2) - 1
    if n_words != (n * (n + 1)):
        raise RuntimeError("Could not infer a square matrix from file")
    return n


def read_fortran_rectangle(
    file: Union[str, FileIO, Path],
    n_columns: int,
    *,
    zones: Union[int, Iterable[int], pd.Index] = None,
    tall: bool = False,
    reindex_rows: bool = False,
    fill_value: Union[int, float] = None,
) -> Union[NDArray, pd.DataFrame, pd.Series]:
    """Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to NOT be square. Also works
    with square matrices.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based
    positional index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns
    present must be known, since the format does not self-specify.

    Args:
        file(str | FileIO | Path): The file to read.
        n_columns (int): The number of columns in the matrix.
        zones (int | Iterable[int] | pandas.Index, optional): Defaults to ``None``. An Index or Iterable will be
            interpreted as the zone labels for the matrix rows and columns; returning a DataFrame or Series (depending
            on `tall`). If an integer is provided, the returned ndarray will be truncated to this 'number of zones'.
        tall (bool, optional): Defaults to ``False``. If true, a 'tall' version of the matrix will be returned.
        reindex_rows (bool, optional): Defaults to ``False``. If true, and zones is an Index, the returned DataFrame
            will be reindexed to fill-in any missing rows.
        fill_value (optional): Defaults to ``None``. The value to pass to ``pandas.reindex()``

    Returns:
        NDArray, pandas.DataFrame or pandas.Series

    Raises:
        ValueError: if the shape is not valid.
    """
    with open_file(file, mode="rb") as reader:
        n_columns = int(n_columns)

        matrix = np.fromfile(reader, dtype=np.float32)
        rows = len(matrix) // (n_columns + 1)
        if len(matrix) != (rows * (n_columns + 1)):
            raise ValueError("Matrix shape is not valid")

        matrix.shape = rows, n_columns + 1

        # Convert binary representation from float to int, then subtract 1 since FORTRAN uses 1-based positional
        # indexing
        row_index = np.frombuffer(matrix[:, 0].tobytes(), dtype=np.int32) - 1
        matrix = matrix[:, 1:]

        if zones is None:
            if tall:
                matrix.shape = matrix.shape[0] * matrix.shape[1]
            return matrix

        if isinstance(zones, (int, np.int_)):
            matrix = matrix[:zones, :zones]

            if tall:
                matrix.shape = zones * zones
            return matrix

        nzones = len(zones)
        matrix = matrix[:nzones, :nzones]
        row_labels = zones.take(row_index[:nzones])
        matrix = pd.DataFrame(matrix, index=row_labels, columns=zones)

        if reindex_rows:
            matrix = matrix.reindex(zones, axis=0, fill_value=fill_value)

        if tall:
            return matrix.stack()
        return matrix


def read_fortran_square(
    file: Union[str, FileIO, Path],
    *,
    zones: Union[int, Iterable[int], pd.Index] = None,
    tall: bool = False,
) -> Union[NDArray, pd.DataFrame, pd.Series]:
    """Reads a FORTRAN-friendly .bin file (a.k.a. 'simple binary format') which is known to be square.

    This file format is an array of 4-bytes, where each row is prefaced by an integer referring to the 1-based
    positional index that FORTRAN uses. The rest of the data are in 4-byte floats. To read this, the number of columns
    present must be known, since the format does not self-specify. This method can infer the shape if it is square.

    Args:
        file (str | FileIO | Path): The file to read.
        zones (int | pandas.Index | Iterable[int], optional): Defaults to ``None``. An Index or Iterable will be
            interpreted as the zone labels for the matrix rows and columns; returning a DataFrame or Series (depending
            on ``tall``). If an integer is provided, the returned ndarray will be truncated to this 'number of zones'.
            Otherwise, the returned ndarray will be size to the maximum number of zone dimensioned by the Emmebank.
        tall (bool, optional): Defaults to ``False``. If True, a 1D data structure will be returned. If ``zone_index``
            is provided, a Series will be returned, otherwise a 1D ndarray.

    Returns:
        NDArray, pandas.DataFrame, or pandas.Series
    """
    with open_file(file, mode="rb") as reader:
        floats = np.fromfile(reader, dtype=np.float32)
        n_words = len(floats)
        matrix_size = _infer_fortran_zones(n_words)
        floats.shape = matrix_size, matrix_size + 1

        data = floats[:, 1:]

        if zones is None:
            if tall:
                n = np.prod(data.shape)
                data.shape = n
                return data
            return data

        if isinstance(zones, (int, np.int_)):
            data = data[:zones, :zones]

            if tall:
                data.shape = zones * zones
                return data
            return data
        elif zones is None:
            return data

        zones = pd.Index(zones)
        n = len(zones)
        data = data[:n, :n]

        matrix = pd.DataFrame(data, index=zones, columns=zones)

        return matrix.stack() if tall else matrix


def to_fortran(
    matrix: Union[NDArray, pd.DataFrame, pd.Series],
    file: Union[str, FileIO, Path],
    *,
    n_columns: int = None,
    min_index: int = 1,
    force_square: bool = True,
) -> None:
    """Writes a FORTRAN-friendly .bin file (a.k.a. 'simple binary format'), in a square format.

    Args:
        matrix (pandas.DataFrame | pandas.Series | NDArray): The matrix to write to disk. If a Series is
            given, it MUST have a MultiIndex with exactly 2 levels to unstack.
        file (str | FileIO | Path): The path or file handler to write to.
        n_columns (int, optional): Defaults to ``None``. Specifies a desired "width" of the matrix file. For example,
            ``n_columns=4000`` on a 3500x3500 matrix will pad the width with 500 extra columns containing 0. If ``None``
            is provided or the value is <= the width of the given matrix, no padding will be performed.
        min_index (int, optional): Defaults to ``1``. The lowest numbered row. Used when slicing matrices
        force_square (bool, optional): Defaults to ``True``.
    """
    if min_index < 1:
        raise ValueError("`min_index` must be >= 1")
    array = coerce_matrix(matrix, force_square=force_square)

    if n_columns is not None and n_columns > array.shape[1]:
        extra_columns = n_columns - array.shape[1]
        array = expand_array(array, extra_columns, axis=1)

    with open_file(file, mode="wb") as writer:
        rows, columns = array.shape
        temp = np.zeros([rows, columns + 1], dtype=np.float32)
        temp[:, 1:] = array

        index = np.arange(min_index, rows + 1, dtype=np.int32)
        # Mask the integer binary representation as floating point
        index_as_float = np.frombuffer(index.tobytes(), dtype=np.float32)
        temp[:, 0] = index_as_float

        temp.tofile(writer)
