from __future__ import annotations

from io import FileIO
from json import loads
from os import PathLike
from typing import Iterable, List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    import h5py
except ImportError:
    h5py = None

try:
    from cityphi.feature import Feature
    from inro.emme.agent import Table
except (ImportError, RuntimeError):
    Feature = None
    Table = None

from .common import coerce_matrix, open_file

# region EMME Matrices


def read_mdf(
    file: Union[str, FileIO, PathLike],
    *,
    raw: bool = False,
    tall: bool = False,
) -> Union[NDArray, pd.DataFrame, pd.Series]:
    """Reads Emme's official matrix "binary serialization" format, created using ``inro.emme.matrix.MatrixData.save()``.
    There is no official extension for this type of file; '.mdf' is recommended. '.emxd' is also sometimes encountered.

    Args:
        file (str | FileIO | PathLike): The file to read.
        raw (bool, optional): Defaults to ``False``. If ``True``, returns an unlabelled ndarray. Otherwise, a DataFrame
            will be returned.
        tall (bool, optional): Defaults to ``False``. If ``True``, a 1D data structure will be returned. If
            ``raw=False``, a Series will be returned, otherwise a 1D ndarray.

    Returns:
        NDArray, pandas.DataFrame, or pandas.Series: The matrix stored in the file.
    """
    with open_file(file, mode="rb") as file_handler:
        magic, version, dtype_index, ndim = np.fromfile(file_handler, np.uint32, count=4)

        if magic != 0xC4D4F1B2 or version != 1 or not (0 < dtype_index <= 4) or not (0 < ndim <= 2):
            raise IOError(
                "Unexpected file header: magic number: %X, version: %d, data type: %d, dimensions: %d."
                % (magic, version, dtype_index, ndim)
            )

        shape = np.fromfile(file_handler, np.uint32, count=ndim)

        index_list = []
        for n_items in shape:
            indices = np.fromfile(file_handler, np.int32, n_items)
            index_list.append(indices)

        dtype = {1: np.float32, 2: np.float64, 3: np.int32, 4: np.uint32}[dtype_index]
        flat_length = shape.prod()  # Multiply the shape tuple
        matrix = np.fromfile(file_handler, dtype, count=flat_length)

        if raw and tall:
            return matrix

        matrix.shape = shape

        if raw:
            return matrix

        if ndim == 1:
            return pd.Series(matrix, index=index_list[0])
        elif ndim == 2:
            matrix = pd.DataFrame(matrix, index=index_list[0], columns=index_list[1])

            return matrix.stack() if tall else matrix

        raise NotImplementedError()  # This should never happen


def to_mdf(
    matrix: Union[pd.DataFrame, pd.Series],
    file: Union[str, FileIO, PathLike],
) -> None:
    """Writes a matrix to Emme's official "binary serialization" format, which can be loaded in Emme using
    ``inro.emme.matrix.MatrixData.load()``. There is no official extension for this type of file; '.mdf' is recommended.

    Args:
        matrix (pandas.DataFrame | panda.Series): The matrix to write to disk. If a Series is given, it MUST have
            a MultiIndex with exactly 2 levels to unstack.
        file (str | FileIO | PathLike): The path or file handler to write to.
    """
    if isinstance(matrix, pd.Series):
        row_index = matrix.index.get_level_values(0).unique()
        column_index = matrix.index.get_level_values(1).unique()
    elif isinstance(matrix, pd.DataFrame):
        row_index = matrix.index
        column_index = matrix.columns
    else:
        raise TypeError("Only labelled matrix objects are supported")

    with open_file(file, mode="wb") as writer:
        data = coerce_matrix(matrix, allow_raw=False)

        np.array([0xC4D4F1B2, 1, 1, 2], dtype=np.uint32).tofile(writer)  # Header
        np.array(data.shape, dtype=np.uint32).tofile(writer)  # Shape

        np.array(row_index, dtype=np.int32).tofile(writer)
        np.array(column_index, dtype=np.int32).tofile(writer)

        data.tofile(writer)


def peek_mdf(
    file: Union[str, FileIO, PathLike],
    *,
    as_index: bool = True,
) -> Union[List[List[int]], List[pd.Index]]:
    """Partially opens an MDF file to get the zone system of its rows and its columns.

    Args:
        file (str | FileIO | PathLike): The file to read.
        as_index (bool, optional): Defaults to ``True``. Set to ``True`` to return a pandas.Index object rather than
            List[int]

    Returns:
        List[int] or List[pandas.Index]: One item for each dimension. If ``as_index=True``, the items will be pandas.Index objects, otherwise they will be List[int]
    """
    with open_file(file, mode="rb") as file_handler:
        magic, version, dtype_index, ndim = np.fromfile(file_handler, np.uint32, count=4)

        if magic != 0xC4D4F1B2 or version != 1 or not (0 < dtype_index <= 4) or not (0 < ndim <= 2):
            raise IOError(
                "Unexpected file header: magic number: %X, version: %d, data type: %d, dimensions: %d."
                % (magic, version, dtype_index, ndim)
            )

        shape = np.fromfile(file_handler, np.uint32, count=ndim)

        index_list = []
        for n_items in shape:
            indices = np.fromfile(file_handler, np.int32, n_items)
            index_list.append(indices)

        if not as_index:
            return index_list

        return [pd.Index(zones) for zones in index_list]


def read_emx(
    file: Union[str, FileIO, PathLike],
    *,
    zones: Union[int, Iterable[int], pd.Index] = None,
    tall: bool = False,
) -> Union[NDArray, pd.DataFrame, pd.Series]:
    """Reads an "internal" Emme matrix (found in `<Emme Project>/Database/emmemat`); with an '.emx' extension. This data
    format does not contain information about zones. Its size is determined by the dimensions of the Emmebank
    (``Emmebank.dimensions['centroids']``), regardless of the number of zones actually used in all scenarios.

    Args:
        file (str | File | PathLike): The file to read.
        zones (int | Iterable[int] | pandas.Index, optional): Defaults to ``None``. An Index or Iterable will be
            interpreted as the zone labels for the matrix rows and columns; returning a DataFrame or Series (depending
            on ``tall``). If an integer is provided, the returned ndarray will be truncated to this 'number of zones'.
            Otherwise, the returned ndarray will be size to the maximum number of zone dimensioned by the Emmebank.
        tall (bool, optional): Defaults to ``False``. If True, a 1D data structure will be returned. If ``zone_index``
            is provided, a Series will be returned, otherwise a 1D ndarray.

    Returns:
        NDArray, pandas.DataFrame, or pandas.Series.

    Examples:
        For a project with 20 zones:

        >>> matrix = read_emx("Database/emmemat/mf1.emx")
        >>> print type(matrix), matrix.shape
        (numpy.ndarray, (20, 20))

        >>> matrix = read_emx("Database/emmemat/mf1.emx", zones=10)
        >>> print type(matrix), matrix.shape
        (numpy.ndarray, (10, 10))

        >>> matrix = read_emx("Database/emmemat/mf1.emx", zones=range(10))
        >>> print type(matrix), matrix.shape
        <class 'pandas.core.frame.DataFrame'> (10, 10)

        >>> matrix = read_emx("Database/emmemat/mf1.emx", zones=range(10), tall=True)
        >>> print type(matrix), matrix.shape
        <class 'pandas.core.series.Series'> 100

    """
    with open_file(file, mode="rb") as reader:
        data = np.fromfile(reader, dtype=np.float32)

        n = int(len(data) ** 0.5)
        if len(data) != n**2:
            raise RuntimeError("Matrix shape from file is not square")

        if zones is None and tall:
            return data

        data.shape = n, n

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


def to_emx(
    matrix: Union[pd.DataFrame, pd.Series, NDArray],
    file: Union[str, FileIO, PathLike],
    emmebank_zones: int,
) -> None:
    """Writes an "internal" Emme matrix (found in `<Emme Project>/Database/emmemat`); with an '.emx' extension. The
    number of zones that the Emmebank is dimensioned for must be known in order for the file to be written correctly.

    Args:
        matrix (pandas.DataFrame | pandas.Series | numpy.ndarray): The matrix to write to disk. If a Series is
            given, it MUST have a MultiIndex with exactly 2 levels to unstack.
        file (str | FileIO | PathLike): The path or file handler to write to.
        emmebank_zones (int): The number of zones the target Emmebank is dimensioned for.
    """
    if emmebank_zones <= 0:
        raise ValueError("Emmebank zones must be > 0")

    with open_file(file, mode="wb") as writer:
        data = coerce_matrix(matrix)
        n = data.shape[0]
        if n > emmebank_zones:
            out = data[:emmebank_zones, :emmebank_zones].astype(np.float32)
        else:
            out = np.zeros([emmebank_zones, emmebank_zones], dtype=np.float32)
            out[:n, :n] = data

        out.tofile(writer)


# endregion

# region Features


if (h5py is None) and (Feature is None) and (Table is None):
    # If neither h5py or OpenPaths Python API is available, gracefully disable the function

    def read_feat(*args, **kwargs):
        raise NotImplementedError()

elif (Feature is None) and (Table is None):
    # If OpenPaths Python API is not available, use h5py to read a basic version of the feature file

    def read_feat(file: Union[str, PathLike]) -> pd.DataFrame:
        """Reads tabular data stored in a feature file produced by OpenPaths applications. Uses h5py.

        Args:
            file (str | PathLike): The file to read

        Returns:
            pd.DataFrame
        """
        retval = []
        with h5py.File(str(file), "r") as f:
            atts = loads(f["attributes"]["data"][()].decode())
            for i, att in enumerate(atts):
                col_name = f"col_{i}"
                if att["dtype"] == "string":
                    v = loads(f["attributes"][col_name][()].decode())
                elif att["is_list"]:
                    continue  # TODO
                else:
                    v = f["attributes"][col_name][()]
                s = pd.Series(v, name=att["name"])
                retval.append(s)

        retval = pd.concat(retval, axis=1)
        if "feature_id" in retval:
            retval.drop("feature_id", axis=1, inplace=True)

        return retval

else:
    # If OpenPaths Python API is available, use it to read a full-featured version of the feature file

    def read_feat(file: Union[str, PathLike]) -> pd.DataFrame:
        """Reads tabular data stored in a feature file produced by OpenPaths applications. Uses the EMME Python API.

        Args:
            file (str | PathLike): The file to read

        Returns:
            pd.DataFrame
        """
        return Table(Feature.from_feature_file(str(file))).to_dataframe()


# endregion
