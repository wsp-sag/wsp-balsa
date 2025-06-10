from __future__ import annotations

from os import PathLike
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..general import sort_nicely
from ..matrices import fast_unstack

try:
    import openmatrix as omx
except ImportError:
    omx = None

MATRIX_TYPES = Union[pd.DataFrame, pd.Series, NDArray]


if omx is None:

    def read_omx(*args, **kwargs):
        raise NotImplementedError()

    def to_omx(*args, **kwargs):
        raise NotImplementedError()

else:

    def read_omx(
        src_fp: Union[str, PathLike],
        *,
        tables: Iterable[str] = None,
        mapping: str = None,
        tall: bool = False,
        raw: bool = False,
        squeeze: bool = True,
    ) -> Union[MATRIX_TYPES, Dict[str, MATRIX_TYPES]]:
        """
        Reads Open Matrix (OMX) files. An OMX file can contain multiple matrices, so this function
        typically returns a Dict.

        Args:
            src_fp (str | PathLike): OMX file from which to read. Cannot be an open file handler.
            tables (Iterable[str], optional): List of matrices to read from the file. If None, all matrices will be
                read.
            mapping (str, optional): The zone number mapping to use, if known in advance.
            tall (bool, optional) : If True, matrices will be returned in 1D format. Otherwise, a 2D object is returned.
            raw (bool, optional): If True, matrices will be returned as raw Numpy arrays. Otherwise, Pandas objects are
                returned
            squeeze (bool, optional): If True, and the file contains exactly one matrix, return that matrix instead of a
                Dict.

        Returns:
            The matrix, or matrices contained in the OMX file.

        """
        omx_file = omx.open_file(str(src_fp), mode="r")
        try:
            table_names: List[str] = sort_nicely(omx_file.list_matrices()) if tables is None else list(tables)

            if not raw:
                if mapping is None:
                    rows, columns = omx_file.shape()
                    if rows != columns:
                        raise NotImplementedError("Handling of non-square matrices not implemented yet")
                    labels = pd.Index(range(rows))
                else:
                    zone_mapping: Dict[int, int] = omx_file.mapping(mapping)
                    labels = pd.Index(zone_mapping.keys())
                if tall:
                    labels = pd.MultiIndex.from_product([labels, labels], names=["o", "d"])

            retval = {}
            for name in table_names:
                matrix = np.array(omx_file[name])
                if tall:
                    n = matrix.shape[0] * matrix.shape[1]
                    matrix.shape = n

                if not raw:
                    if tall:
                        matrix = pd.Series(matrix, index=labels, name=name)
                    else:
                        matrix = pd.DataFrame(matrix, index=labels, columns=labels)

                retval[name] = matrix

            if (len(retval) == 1) and squeeze:
                return retval[table_names[0]]
            return retval
        finally:
            omx_file.close()

    def to_omx(
        dst_fp: Union[str, PathLike],
        tables: Dict[str, MATRIX_TYPES],
        *,
        zone_index: pd.Index = None,
        title: str = "",
        descriptions: Dict[str, str] = None,
        attrs: Dict[str, Dict] = None,
        mapping_name: str = "zone_numbers",
    ):
        """Creates a new (or overwrites an old) OMX file with a collection of matrices.

        Args:
            dst_fp (str | PathLike): OMX to write.
            tables (Dict[str, pd.DataFrame | pd.Series | np.ndarray]: Collection of matrices to write. MUST be a dict,
                to permit the encoding of matrix metadata, and must contain the same types: all Numpy arrays, all
                Series, or all DataFrames. Checking is done to ensure that all items have the same shape and labels.
            zone_index: (pd.Index, optional): Override zone labels to use. Generally only useful if writing a dict of
                raw Numpy arrays.
            title (str, optional): The title saved in the OMX file.
            descriptions (Dict[str, str], optional): A dict of descriptions (one for each given matrix).
            attrs (Dict[str, Dict], optional): A dict of dicts (one for each given matrix).
            mapping_name (str, optional): Name of the mapping internal to the OMX file
        """

        matrices, zone_index = _prep_matrix_dict(tables, zone_index)

        if descriptions is None:
            descriptions = {name: "" for name in matrices.keys()}
        if attrs is None:
            attrs = {name: None for name in matrices.keys()}

        omx_file = omx.open_file(str(dst_fp), mode="w", title=title)
        try:
            omx_file.create_mapping(mapping_name, zone_index.tolist())
            for name, array in matrices.items():
                description = descriptions[name]
                attr = attrs[name]
                omx_file.create_matrix(name, obj=np.ascontiguousarray(array), title=description, attrs=attr)
        finally:
            omx_file.close()

    def _prep_matrix_dict(
        matrices: Dict[str, MATRIX_TYPES], desired_zone_index: pd.Index
    ) -> Tuple[Dict[str, np.ndarray], pd.Index]:
        collection_type = _check_types(matrices)

        if collection_type == "RAW":
            checked, n = _check_raw_matrices(matrices)
            zone_index = pd.Index(range(n))
        elif collection_type == "SERIES":
            checked, zone_index = _check_matrix_series(matrices)
        elif collection_type == "FRAME":
            checked, zone_index = _check_matrix_frames(matrices)
        else:
            raise NotImplementedError(collection_type)

        if desired_zone_index is not None:
            assert desired_zone_index.equals(zone_index)

        return checked, zone_index

    def _check_types(matrices: Dict[str, MATRIX_TYPES]) -> str:
        gen = iter(matrices.values())
        first = next(gen)

        item_type = "RAW"
        if isinstance(first, pd.Series):
            item_type = "SERIES"
        elif isinstance(first, pd.DataFrame):
            item_type = "FRAME"

        msg = "All items must be the same type"

        for item in gen:
            if item_type == "FRAME":
                assert isinstance(item, pd.DataFrame), msg
            elif item_type == "SERIES":
                assert isinstance(item, pd.Series), msg
            else:
                assert isinstance(item, np.ndarray), msg

        return item_type

    def _check_raw_matrices(matrices: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int]:
        gen = iter(matrices.items())
        name, matrix = next(gen)

        n_dim = len(matrix.shape)
        if n_dim == 1:
            shape = matrix.shape[0]
            n = int(shape**0.5)
            assert n * n == shape, "Only tall matrices that decompose to square shapes are permitted."
            matrix = matrix[...]
            matrix.shape = n, n
        elif n_dim == 2:
            n, cols = matrix.shape
            assert n == cols, "Only square matrices are permitted"
        else:
            raise TypeError("Only 1D and 2D arrays can be saved to OMX files")

        retval = {name: matrix}
        for name, matrix in gen:
            assert len(matrix.shape) == n_dim

            if n_dim == 1:
                assert matrix.shape[0] == (n * n)
                matrix = matrix[...]
                matrix.shape = n, n
            else:
                assert matrix.shape == (n, n)

            retval[name] = matrix

        return retval, n

    def _check_matrix_series(matrices: Dict[str, pd.Series]) -> Tuple[Dict[str, np.ndarray], pd.Index]:
        gen = iter(matrices.items())
        name, matrix = next(gen)

        tall_index = matrix.index
        assert tall_index.nlevels == 2
        matrix = matrix.unstack()
        zone_index = matrix.index
        assert zone_index.equals(matrix.columns)

        retval = {name: matrix.to_numpy()}
        for name, matrix in gen:
            assert tall_index.equals(matrix.index)
            matrix = fast_unstack(matrix, zone_index, zone_index).to_numpy()
            retval[name] = matrix

        return retval, zone_index

    def _check_matrix_frames(matrices: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], pd.Index]:
        gen = iter(matrices.items())
        name, matrix = next(gen)

        zone_index = matrix.index
        assert zone_index.equals(matrix.columns)

        retval = {name: matrix.to_numpy()}
        for name, matrix in gen:
            assert zone_index.equals(matrix.index)
            assert zone_index.equals(matrix.columns)
            retval[name] = matrix.to_numpy()

        return retval, zone_index
