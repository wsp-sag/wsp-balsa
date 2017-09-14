from __future__ import division, absolute_import, print_function, unicode_literals

import pandas as pd
import numpy as np
import numexpr as ne
from six import iteritems

from .expressions import SimpleUsage, DictLiteral, AttributedUsage, LinkedFrameUsage
from ..ldf import LinkedDataFrame
from .parsing import NAN_STR


class ScopeOrientationError(IndexError):
    pass


class ScalarSymbol(object):

    def __init__(self, value):
        self._value = value

    def get_value(self, usage):
        return self._value


class Array1DSymbol(object):

    def __init__(self, array, orientation):
        self._data = array
        self._orientation = orientation

    def get_value(self, usage):
        converted = convert_series(self._data, allow_raw=True)

        length = len(converted)
        new_shape = [1, 1]
        new_shape[self._orientation] = length

        view = converted[...]  # Make a shallow copy
        view.shape = new_shape
        return view


class Array2DSymbol(object):

    def __init__(self, array, orientation):
        self._data = np.transpose(array) if orientation == 1 else array[...]

    def get_value(self, usage):
        return self._data


class FrameSymbol(object):

    def __init__(self, frame, orientation):
        self._frame = frame
        self._orientation = orientation

    def get_value(self, usage):
        series = self._frame[usage.attribute]
        data = convert_series(series)

        length = len(self._frame)
        shape_tuple = [1, 1]
        shape_tuple[self._orientation] = length

        data = data[...]  # Make a shallow copy
        data.shape = shape_tuple
        return data


class LinkedFrameSymbol(object):
    def __init__(self, frame, orientation):
        self._frame = frame
        self._orientation = orientation

    def get_value(self, usage):
        if isinstance(usage, AttributedUsage):
            series = self.__get_attributed_value(usage)
        elif isinstance(usage, LinkedFrameUsage):
            series = self.__get_linked_usage(usage)
        else:
            raise NotImplementedError("This should never happen")

        data = convert_series(series)[...]  # Make a shallow copy so we can change the shape
        n = len(series)
        new_shape = [1, 1]
        new_shape[self._orientation] = n
        data.shape = new_shape

        return data

    def __get_attributed_value(self, usage):
        return self._frame[usage.attribute]

    def __get_linked_usage(self, usage):
        item = self._frame
        for attribute in reversed(usage.stack):
            item = getattr(item, attribute)

        if usage.func is not None:
            method = getattr(item, usage.func)
            func_expr = "1" if usage.func_expr is None else usage.func_expr
            item = method(func_expr)

        if not isinstance(item, pd.Series):
            pretty_stack = '.'.join(str(c) for c in reversed(usage.stack))
            raise AttributeError("Chained lookup '%s' does not point to a valid Series" % pretty_stack)

        return item


class PanelSymbol(object):

    def __init__(self, panel):
        self._data = panel

    def get_value(self, usage):
        raise NotImplementedError()


class Scope(object):

    def __init__(self, model):
        self._root = model
        self._empty_symbols = None
        self._filled_symbols = None

        self._records = None
        self._alternatives = None

    def set_record_index(self, index):
        """
        Manually set the index of records being processed.

        If some symbols are already filled, calling this method CLEARS their data, resetting them back to 'empty'.

        Args:
            index (List or Index): Any sequence that Pandas accepts when constructing an Index.

        """
        self._initialize()
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        self._records = index

    def empty_symbols(self):
        """
        Iterate through the set of empty (not yet filled) symbols.

        Returns:
            Iterable[str]: Symbols that are empty (have no set data just yet)

        """
        raise NotImplementedError()

    def fill_symbol(self, symbol_name, data, orientation=None, strict=True, allow_unused=True):
        """
        Associate an empty symbol with actual data, usually ararys, DataFrames, and/or Series. Symbol usages are
        collected from the list of model expressions already-loaded, so the type of `data` must conform to the rules of
        usage.

        Args:
            symbol_name (str): The name of the symbol being filled.
            data: Numerical data to be associated with the filled symbol. Supported types include scalar numbers, 1-D
                arrays, 2-D arrays, Series, DataFrames, and Dictionaries of DataFrames / Panels. Not all data types are
                supported by all use cases, refer to usage rules for further details.
            orientation (int): Specifies which dimension of the utility table to which the FIRST dimension (axis)
                of the data is oriented: 0: oriented to the records, 1: oriented to the alternatives. This is useful in
                cases where the records index and alternatives index are the same (the labels are therefore ambiguous),
                or the same length (for adding unlabelled data).
            strict (bool): In the case that Cheval is unable to recognize or support the type of `data`, turning off
                `strict` will permit the data to be filled at this time, and checked by the NumExpr engine later. In
                most cases, this should probably be left to True.
            allow_unused (bool): Disables raising KeyError if the symbol is not found in the expressions. This can be a
                common occurrence if the expressions are read in from user input and the code makes more symbols
                available for computation than are used.

        Raises:
            KeyError: If the symbol_name is not found used in any Expression and allow_unused=False
            TypeError: If the type of `data` was not understood or not supported by the scoping rules.
            ScopeOrientationError: If the data do not conform to the rules for scoping.

        """
        self._initialize()

        if symbol_name not in self._empty_symbols:
            if allow_unused:
                return  # Do not actually add the symbol, since it is known to be unused
            else:
                raise KeyError("Symbol '%s' is not used in any expression" % symbol_name)

        symbol_usage = self._empty_symbols[symbol_name]

        if symbol_usage is LinkedFrameUsage:
            symbol_meta = self._fill_linked(data, orientation)
        elif symbol_usage is AttributedUsage:
            symbol_meta = self._fill_attributed(data, orientation)
        elif symbol_usage is DictLiteral:
            symbol_meta = self._fill_simple(symbol_usage.series, orientation=1)
        elif symbol_usage is SimpleUsage:
            symbol_meta = self._fill_simple(data, orientation, strict)
        else:
            raise NotImplementedError("Usage type '%s' not understood" % symbol_usage)

        del self._empty_symbols[symbol_name]
        self._filled_symbols[symbol_name] = symbol_meta

    def _compute_utilities(self, n_threads, logger=None):
        assert len(self._empty_symbols) == 0

        model = self._root

        # Allocate an empty utility table
        shape = len(self._records), len(model.tree.node_index)
        utility_table = np.zeros(shape, dtype=np.float64, order='C')

        ne.set_num_threads(n_threads)

        # Evaluate each expression
        for expr in model.expressions:
            try:
                self._evaluate_single_expression(expr, utility_table)
            except Exception as e:
                if logger is not None:
                    logger.error("Error while evaluating '%s'" % expr._raw_expr)
                raise e

        return utility_table

    def _evaluate_single_expression(self, expr, utility_table):
        # Setup local dictionary of data
        local_dict = {NAN_STR: np.nan}
        for symbol_name, list_of_usages in expr.symbols():
            if isinstance(list_of_usages, DictLiteral):
                list_of_usages = [list_of_usages]

            for usage in list_of_usages:
                # Usage is one of SimpleUsage, DictLiteral, AttributedUsage, or LinkedFrameUsage

                # Symbol meta is an instance of scope.AbstractSymbol
                symbol_meta = self._filled_symbols[symbol_name]
                data = symbol_meta.get_value(usage)

                if isinstance(usage, SimpleUsage):
                    # In this case, no substitution was performed, so we can just use the symbol name
                    local_dict[symbol_name] = data
                else:
                    # Otherwise, we need to set the data to another alias
                    local_dict[usage.substitution] = data

        # Run the expression.
        final_expression = '__out + (%s)' % expr._parsed_expr
        local_dict['__out'] = utility_table
        ne.evaluate(final_expression, local_dict=local_dict, out=utility_table)

    def _fill_simple(self, data, orientation=None, strict=True):
        """"""

        '''
        This is a bit ugly and should probably be broken out into separate functions for different types
        '''

        if isinstance(data, np.ndarray):
            # Unlabelled 1- or 2-D array
            self._check_records()

            if data.ndim == 1:
                # 1-D array
                n_rows = len(data)
                n_rec = len(self._records)
                n_alts = len(self._alternatives)
                if n_rec == n_alts:
                    # Orientation matters
                    if n_rows != n_rec:
                        raise ScopeOrientationError("1D data array shape incompatible with records or alternatives")
                    if orientation is None:
                        raise ScopeOrientationError("Orientation must be provided for arrays when length of records "
                                                    "matches length of alternatives")
                    return Array1DSymbol(data, orientation)
                else:
                    # Infer orientation from length of array
                    if n_rows == n_rec:
                        # Oriented to the records
                        return Array1DSymbol(data, 0)
                    elif n_rows == n_alts:
                        return Array1DSymbol(data, 1)
                    else:
                        raise ScopeOrientationError("1D arrays must be as long as either the records or alternatives")
            elif data.ndim == 2:
                # 2-D array
                n_rows, n_cols = data.shape
                n_rec, n_alts = len(self._records), len(self._alternatives)

                if n_rec == n_alts:
                    # Orientation matters
                    if n_rows != n_rec or n_cols != n_alts:
                        raise ScopeOrientationError("2D data array shape incompatible with records and alternatives")
                    if orientation is None:
                        raise ScopeOrientationError("Orientation must be provided for arrays when length of records "
                                                    "matches length of alternatives")
                    return Array2DSymbol(data, orientation)
                elif n_rows == n_rec and n_cols == n_alts:
                    return Array2DSymbol(data, 0)
                elif n_rows == n_alts and n_cols == n_rec:
                    return Array2DSymbol(data, 1)
                else:
                    raise ScopeOrientationError("2D array shapes must align with both the records and alternatives")
            else:
                raise ScopeOrientationError("Numpy arrays are permitted, but only with 1 or 2 dimensions (found more)")
        elif isinstance(data, pd.DataFrame):
            # Labelled 2-D array
            self._check_records()
            # For a simple symbols, this is only permitted if BOTH axes align with the utilities table
            if self._records.equals(self._alternatives):
                # Orientation is ambiguous
                self._check_orientation(orientation)
                if not data.index.equals(self._records) or not data.columns.equals(self._records):
                    raise ScopeOrientationError("Simple DataFrames must align with both the records and alternatives")
                return Array2DSymbol(data.values, orientation)
            elif data.index.equals(self._records) and data.columns.equals(self._alternatives):
                return Array2DSymbol(data.values, 0)
            elif data.index.equals(self._alternatives) and data.columns.equals(self._records):
                return Array2DSymbol(data.values, 1)
            else:
                raise ScopeOrientationError("Simple DataFrames must align with both the records and alternatives")
        elif isinstance(data, pd.Series):
            # Labelled 1-D array
            self._check_records()

            if self._records.equals(self._alternatives):
                # Orientation is ambiguous
                self._check_orientation(orientation)
                if not data.index.equals(self._records):
                    raise ScopeOrientationError("Series must align with either the records or alternatives")

                return Array1DSymbol(data, orientation)
            elif data.index.equals(self._records):
                return Array1DSymbol(data, 0)
            elif data.index.equals(self._alternatives):
                return Array1DSymbol(data, 1)
            else:
                raise ScopeOrientationError("Series must align with either the records or the alternatives")
        elif strict:
            # Rigourous type-checking for scalars if strict is required
            if not isinstance(data, (int, float, np.int_, np.float_)):
                raise TypeError("Unsupported simple symbol data type: %s" % type(data))
            return ScalarSymbol(data)
        else:
            # Permissive; let NumExpr figure out if the data is valid
            return ScalarSymbol(data)

    @staticmethod
    def _check_orientation(orientation):
        if orientation is None:
            raise ScopeOrientationError("Orientation must be provided when the record index exactly equals the"
                                        " alternatives index.")
        if orientation != 0 and orientation != 1:
            raise ValueError("Orientation must be either 0 or 1")

    def _check_records(self):
        assert self._records is not None, "This operation is not allowed if the records have not been set."

    def _fill_attributed(self, data, orientation=None):
        self._check_records()

        if isinstance(data, pd.DataFrame):
            if self._records.equals(self._alternatives):
                self._check_orientation(orientation)
                if not data.index.equals(self._records):
                    raise ScopeOrientationError("Filling attributed usage with a DataFrame is permitted, but the index "
                                                "must align with either the records or the alternatives")
                return FrameSymbol(data, orientation)
            elif data.index.equals(self._records):
                return FrameSymbol(data, 0)
            elif data.index.equals(self._alternatives):
                return FrameSymbol(data, 1)
            else:
                raise ScopeOrientationError("Filling attributed usage with a DataFrame is permitted, but the index "
                                            "must align with either the records or the alternatives")

        elif isinstance(data, (pd.Panel, dict)):
            if isinstance(data, dict):
                data = pd.Panel(dict)

            if data.major_axis.equals(self._alternatives) and data.minor_axis.equals(self._records):
                data = data.transpose('items', 'minor_axis', 'major_axis')
            elif not (data.major_axis.equals(self._records) and data.minor_axis.equals(self._alternatives)):
                raise ScopeOrientationError("Panel symbols major and minor axes must align with the records and "
                                            "alternatives")
            return PanelSymbol(data)

        else:
            raise TypeError("Only DataFrames, dictionaries of DataFrames, or Panels can fill attributed symbols")

    def _fill_linked(self, data, orientation):
        assert isinstance(data, LinkedDataFrame), "Only LinkedDataFrames can fill linked symbols."
        self._check_records()

        if self._records.equals(self._alternatives):
            self._check_orientation(orientation)

            if not data.index.equals(self._records):
                raise ScopeOrientationError("Filling linked usage with a LinkedDataFrame is permitted, but the index "
                                            "must align with either the records or alternatives")

            return LinkedFrameSymbol(data, orientation)
        elif data.index.equals(self._records):
            return LinkedFrameSymbol(data, 0)
        elif data.index.equals(self._alternatives):
            return LinkedFrameSymbol(data, 1)
        else:
            raise ScopeOrientationError("Filling linked usage with a LinkedDataFrame is permitted, but the index must "
                                        "align with either the records or alternatives")

    def clear(self):
        self._empty_symbols = None
        self._filled_symbols = None
        self._records = None
        self._alternatives = None

    def _initialize(self):
        if self._empty_symbols is None or self._filled_symbols is None:
            self._empty_symbols, dict_literals = self._root.expressions.get_symbols()
            self._filled_symbols = {}
            self._alternatives = self._root.tree.node_index

            self._fill_dict_literals(dict_literals)

    def _fill_dict_literals(self, dict_literals):
        for alias, usage in iteritems(dict_literals):
            expanded_series = usage.series.reindex(self._alternatives, fill_value=0.0)
            symbol = Array1DSymbol(expanded_series.values, orientation=1)
            self._filled_symbols[alias] = symbol

    def _symbolize(self):
        if self._empty_symbols:
            raise AttributeError("Cannot evaluate expressions when there are still empty symbols that need to be "
                                 "filled")
        return self._filled_symbols


def convert_series(s, allow_raw=False):
    dtype = s.dtype

    if dtype.name == 'category':
        # Categorical column
        categorical = s.values

        category_index = categorical.categories
        if category_index.dtype.name == 'object':
            max_len = category_index.str.len().max()
            typename = 'S%s' % max_len
        else:
            typename = category_index.dtype

        return categorical.astype(typename)
    elif dtype.name == 'object':
        # Object or text column
        max_length = s.str.len().max()
        if np.isnan(max_length):
            raise TypeError("Could not get max string length")

        return s.values.astype("S%s" % max_length)
    elif np.issubdtype(dtype, np.datetime64):
        raise TypeError("Datetime columns are not supported")
    elif np.issubdtype(dtype, np.timedelta64):
        raise TypeError("Timedelta columns are not supported")
    try:
        return s.values
    except:
        if allow_raw: return s
        raise
