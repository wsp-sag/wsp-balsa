from __future__ import annotations

import re
import zipfile
from os import PathLike
from pathlib import Path
from typing import Hashable, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

EMME_ENG_UNITS = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 0.001, "k": 1000.0, "M": 1e6, "G": 1e9, "T": 1e12}


def parse_tmg_ncs_line_id(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """A function to parse line IDs based on TMG Network Coding Standard conventions. Returns pandas Series objects
    corresponding to the parsed operator and route IDs"""
    operator = s.str[:2].str.replace(r"\d+", "", regex=True)

    route = s.str.replace(r"\D", "", regex=True).str.lstrip("0")  # Isolate for route number, if applicable
    for idx, _ in route[route == ""].items():  # If no route number, assume route id based on TMG NCS convention
        route.loc[idx] = s.loc[idx][len(operator.loc[idx]) : -1]

    return operator, route


def process_emme_eng_notation_series(s: pd.Series, *, to_dtype=float) -> pd.Series:  # TODO: create generic version...
    """A function to convert Pandas Series containing values in Emme's engineering notation"""
    values = s.str.replace(r"\D+", ".", regex=True).astype(to_dtype)
    units = s.str.replace(r"[\d,.]+", "", regex=True).map(EMME_ENG_UNITS).fillna(1.0)
    return values * units


def read_nwp_base_network(
    nwp_fp: Union[str, PathLike],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the base network from a Network Package file (exported from Emme using the TMG Toolbox) into
    DataFrames.

    Args:
        nwp_fp (str | PathLike): File path to the network package.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the nodes and links
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    header_nodes, header_links, last_line = None, None, None
    with zipfile.ZipFile(nwp_fp) as zf:
        for i, line in enumerate(zf.open("base.211"), start=1):
            line = line.strip().decode("utf-8")
            if line.startswith("c"):
                continue  # Skip comment lines
            if line.startswith("t nodes"):
                header_nodes = i
            elif line.startswith("t links"):
                header_links = i
        last_line = i

        # Read nodes
        n_rows = header_links - header_nodes - 2
        data_types = {
            "c": str,
            "Node": np.int64,
            "X-coord": float,
            "Y-coord": float,
            "Data1": float,
            "Data2": float,
            "Data3": float,
            "Label": str,
        }
        nodes = pd.read_csv(
            zf.open("base.211"),
            index_col="Node",
            dtype=data_types,
            skiprows=header_nodes,
            nrows=n_rows,
            delim_whitespace=True,
        )
        nodes.columns = nodes.columns.str.lower()
        nodes.columns = nodes.columns.str.strip()
        nodes.index.name = "node"
        nodes.rename(columns={"x-coord": "x", "y-coord": "y"}, inplace=True)
        nodes["is_centroid"] = nodes["c"] == "a*"
        nodes.drop("c", axis=1, inplace=True)

        # Read links
        n_rows = last_line - header_links - 1
        links = pd.read_csv(
            zf.open("base.211"),
            index_col=["From", "To"],
            skiprows=header_links,
            nrows=n_rows,
            delim_whitespace=True,
            low_memory=False,
        )
        links.columns = links.columns.str.lower()
        links.columns = links.columns.str.strip()
        links.index.names = ["inode", "jnode"]
        mask_mod = links["c"] == "m"
        n_modified_links = len(links[mask_mod])
        if n_modified_links > 0:
            print(f"Ignored {n_modified_links} modification records in the links table")
        links = links[~mask_mod].drop("c", axis=1)
        if "typ" in links.columns:
            links.rename(columns={"typ": "type"}, inplace=True)
        if "lan" in links.columns:
            links.rename(columns={"lan": "lanes"}, inplace=True)

        # Data type conversion
        links = links.astype(
            {"modes": str, "type": int, "lanes": int, "vdf": int}
        )  # simple type casting for non-float
        for col in ["length", "data1", "data2", "data3"]:
            if is_string_dtype(links[col]):  # these columns are usually string if values use Emme engineering notation
                links[col] = process_emme_eng_notation_series(links[col])
            else:
                links[col] = links[col].astype(float)

    return nodes, links


def read_nwp_exatts_list(
    nwp_fp: Union[str, PathLike],
    **kwargs,
) -> pd.DataFrame:
    """A function to read the extra attributes present in a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        **kwargs: Any valid keyword arguments used by ``pandas.read_csv()``.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    kwargs["index_col"] = False
    if "quotechar" not in kwargs:
        kwargs["quotechar"] = "'"

    with zipfile.ZipFile(nwp_fp) as zf:
        df: pd.DataFrame = pd.read_csv(zf.open("exatts.241"), **kwargs)
        df.columns = df.columns.str.strip()
        df["type"] = df["type"].astype("category")

    return df


def _base_read_nwp_att_data(
    nwp_fp: Union[str, PathLike],
    att_type: str,
    index_col: Union[str, List[str]],
    attributes: Union[str, List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    if attributes is not None:
        if isinstance(attributes, Hashable):
            attributes = [attributes]
        elif isinstance(attributes, list):
            pass
        else:
            raise RuntimeError

    if "quotechar" not in kwargs:
        kwargs["quotechar"] = "'"

    with zipfile.ZipFile(nwp_fp) as zf:
        df: pd.DataFrame = pd.read_csv(zf.open(f"exatt_{att_type}.241"), **kwargs)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if is_string_dtype(df[col]):
                df[col] = df[col].str.strip()
        df.set_index(index_col, inplace=True)

    if attributes is not None:
        df = df[attributes].copy()

    return df


def read_nwp_node_attributes(
    nwp_fp: Union[str, PathLike],
    *,
    attributes: Union[str, List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """A function to read node attributes from a Network Package file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        attributes (str | List[str], optional): Defaults to ``None``. Names of node attributes to extract. Note
            that ``'inode'`` will be included by default.
        **kwargs: Any valid keyword arguments used by ``pandas.read_csv()``.

    Returns:
        pd.DataFrame
    """
    return _base_read_nwp_att_data(nwp_fp, "nodes", "inode", attributes, **kwargs)


def read_nwp_link_attributes(
    nwp_fp: Union[str, PathLike],
    *,
    attributes: Union[str, List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """A function to read link attributes from a Network Package file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        attributes (str | List[str], optional): Defaults to ``None``. Names of link attributes to extract. Note
            that ``'inode'`` and ``'jnode'`` will be included by default.
        **kwargs: Any valid keyword arguments used by ``pandas.read_csv()``.

    Returns:
        pd.DataFrame
    """
    return _base_read_nwp_att_data(nwp_fp, "links", ["inode", "jnode"], attributes, **kwargs)


def read_nwp_transit_line_attributes(
    nwp_fp: Union[str, PathLike],
    *,
    attributes: Union[str, List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """A function to read transit line attributes from a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        attributes (str | List[str], optional): Defaults to ``None``. Names of transit line attributes to extract.
            Note that ``'line'`` will be included by default.
        **kwargs: Any valid keyword arguments used by ``pandas.read_csv()``.

    Returns:
        pd.DataFrame
    """
    return _base_read_nwp_att_data(nwp_fp, "transit_lines", "line", attributes, **kwargs)


def read_nwp_traffic_results(
    nwp_fp: Union[str, PathLike],
) -> pd.DataFrame:
    """A function to read the traffic assignment results from a Network Package file (exported from Emme using the TMG
    Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    with zipfile.ZipFile(nwp_fp) as zf:
        df = pd.read_csv(zf.open("link_results.csv"), index_col=["i", "j"])
        df.index.names = ["inode", "jnode"]

    return df


def read_nwp_traffic_results_at_countpost(
    nwp_fp: Union[str, PathLike],
    countpost_att: str,
) -> pd.DataFrame:
    """A function to read the traffic assignment results at countposts from a Network Package file (exported from Emme
    using the TMG Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        countpost_att (str): The name of the extra link attribute containing countpost identifiers. Results will be
            filtered using this attribute.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    if not countpost_att.startswith("@"):
        countpost_att = f"@{countpost_att}"

    countpost_links = read_nwp_link_attributes(nwp_fp, attributes=countpost_att)
    countpost_links = countpost_links[countpost_links[countpost_att] > 0]

    results = read_nwp_traffic_results(nwp_fp)
    results = results[results.index.isin(countpost_links.index)].copy()

    return results


def read_nwp_transit_network(
    nwp_fp: Union[str, PathLike],
    *,
    parse_line_id: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function to read the transit network from a Network Package file (exported from Emme using the TMG Toolbox)
    into DataFrames.

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        parse_line_id (bool, optional): Defaults to ``False``. Option to parse operator and route IDs from line IDs.
            Please note that transit line IDs must adhere to the TMG NCS16 for this option to work properly.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of DataFrames containing the transt lines and segments.
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    # Parse through transit line transaction file
    seg_cols = ["inode", "dwt", "ttf", "us1", "us2", "us3"]
    transit_lines = []
    transit_segs = []
    current_tline = None
    with zipfile.ZipFile(nwp_fp) as zf:
        for line in zf.open("transit.221"):
            line = line.strip().decode("utf-8")
            if line.startswith("c") or line.startswith("t") or line.startswith("path"):
                continue  # Skip
            elif line.startswith("a"):
                parts = re.sub(r"\s+", " ", line.replace("'", " ")).split(" ")
                parts = (
                    parts[1:6] + [" ".join(parts[6:-3])] + parts[-3:]
                )  # reconstruct parts with a joined description
                transit_lines.append(parts)
                current_tline = parts[0]
            else:
                parts = re.sub(r"\s+", " ", line).split(" ")
                if len(parts) < len(seg_cols):
                    parts = parts + [np.nan] * (len(seg_cols) - len(parts))  # row needed for node... pad with NaNs
                parts.insert(0, current_tline)
                transit_segs.append(parts)

    # Create transit segment dataframe
    transit_segs = pd.DataFrame(transit_segs, columns=["line"] + seg_cols)
    transit_segs["inode"] = transit_segs["inode"].astype(np.int64)
    transit_segs["jnode"] = transit_segs.groupby("line")["inode"].shift(-1).fillna(0).astype(np.int64)
    transit_segs["seg_seq"] = (transit_segs.groupby("line").cumcount() + 1).astype(int)
    transit_segs["loop"] = (transit_segs.groupby(["line", "inode", "jnode"])["seg_seq"].cumcount() + 1).astype(int)
    transit_segs.dropna(inplace=True)  # remove rows without dwt, ttf, us1, us2, us3 data (i.e. the padded rows)
    transit_segs = transit_segs[
        ["line", "inode", "jnode", "seg_seq", "loop", "dwt", "ttf", "us1", "us2", "us3"]
    ].copy()
    transit_segs["dwt"] = transit_segs["dwt"].str.replace("dwt=", "", regex=False)
    transit_segs["ttf"] = transit_segs["ttf"].str.replace("ttf=", "", regex=False).astype(np.int16)
    transit_segs["us1"] = transit_segs["us1"].str.replace("us1=", "", regex=False).astype(float)
    transit_segs["us2"] = transit_segs["us2"].str.replace("us2=", "", regex=False).astype(float)
    transit_segs["us3"] = transit_segs["us3"].str.replace("us3=", "", regex=False).astype(float)

    # Create transit lines dataframe
    columns = ["line", "mode", "veh", "headway", "speed", "description", "data1", "data2", "data3"]
    data_types = {  # remember that python 3.6 doesn't guarentee that order is maintained...
        "line": str,
        "mode": str,
        "veh": int,
        "headway": float,
        "speed": float,
        "description": str,
        "data1": float,
        "data2": float,
        "data3": float,
    }
    transit_lines = pd.DataFrame(transit_lines, columns=columns).astype(data_types)
    transit_lines = transit_lines.set_index(["line", "description"]).reset_index()
    if parse_line_id:
        operator, route = parse_tmg_ncs_line_id(transit_lines["line"])
        transit_lines.insert(1, "operator", operator)
        transit_lines.insert(2, "route", route)

    return transit_lines, transit_segs


def read_nwp_transit_result_summary(
    nwp_fp: Union[str, PathLike],
    *,
    parse_line_id: bool = False,
) -> pd.DataFrame:
    """A function to read and summarize the transit assignment boardings and max volumes from a Network Package file
    (exported from Emme using the TMG Toolbox) by operator and route.

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        parse_line_id (bool, optional): Defaults to ``False``. Option to parse operator and route IDs from line IDs.
            Please note that transit line IDs must adhere to the TMG NCS16 for this option to work properly.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    with zipfile.ZipFile(nwp_fp) as zf:
        data_types = {"line": str, "transit_boardings": float, "transit_volume": float}
        df = pd.read_csv(zf.open("segment_results.csv"), usecols=data_types.keys(), dtype=data_types)
        if parse_line_id:
            operator, route = parse_tmg_ncs_line_id(df["line"])
            df["operator"] = operator
            df["route"] = route.str.zfill(route.str.len().max())  # Pad with 0s for groupby sorting purposes
            df = df.groupby(["operator", "route"], as_index=False).agg(
                {"transit_boardings": "sum", "transit_volume": "sum"}
            )
            df["route"] = df["route"].str.lstrip("0")  # Remove 0s padding
            df.set_index(["operator", "route"], inplace=True)
        else:
            df.set_index("line", inplace=True)
        df.rename(columns={"transit_boardings": "boardings", "transit_volume": "total_volume"}, inplace=True)

    return df


def read_nwp_transit_station_results(
    nwp_fp: Union[str, PathLike],
    station_line_nodes: List[int],
) -> pd.DataFrame:
    """A function to read and summarize the transit boardings (on) and alightings (offs) at stations from a Network
    Package file (exported from Emme using the TMG Toolbox).

    Note:
        Ensure that station nodes being specified are on the transit line itself and are not station centroids.

    Args:
        nwp_fp (str | PathLike): File path to the network package.
        station_line_nodes (List[int]): List of transit line nodes representing transit stops/stations

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    with zipfile.ZipFile(nwp_fp) as zf:
        results = pd.read_csv(zf.open("aux_transit_results.csv"), index_col=["i", "j"]).squeeze("columns")

    station_results = pd.DataFrame(index=sorted(station_line_nodes))
    station_results.index.name = "stn_node"

    mask_on = results.index.get_level_values(1).isin(station_line_nodes)
    station_results["on"] = results.loc[mask_on].groupby(level=1).sum().round(3)

    mask_off = results.index.get_level_values(0).isin(station_line_nodes)
    station_results["off"] = results.loc[mask_off].groupby(level=0).sum().round(3)

    return station_results


def read_nwp_transit_segment_results(
    nwp_fp: Union[str, PathLike],
) -> pd.DataFrame:
    """A function to read and summarize the transit segment boardings, alightings, and volumes from a Network Package
    file (exported from Emme using the TMG Toolbox).

    Args:
        nwp_fp (str | PathLike): File path to the network package.

    Returns:
        pd.DataFrame
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    _, segments = read_nwp_transit_network(nwp_fp)
    segments.set_index(["line", "inode", "jnode", "loop"], inplace=True)

    with zipfile.ZipFile(nwp_fp) as zf:
        results = pd.read_csv(zf.open("segment_results.csv"), index_col=["line", "i", "j", "loop"])

    segments["boardings"] = results["transit_boardings"].round(3)
    segments["volume"] = results["transit_volume"].round(3)
    n_missing_segments = len(segments[segments["boardings"].isnull()])
    if n_missing_segments > 0:
        print(f"Found {n_missing_segments} segments with missing results; their results will be set to 0")
        segments.fillna(0, inplace=True)
    segments.reset_index(inplace=True)

    segments["prev_seg_volume"] = segments.groupby("line")["volume"].shift(1).fillna(0)
    segments["alightings"] = segments.eval("prev_seg_volume + boardings - volume").round(3)

    segments.drop(["dwt", "ttf", "us1", "us2", "us3", "prev_seg_volume"], axis=1, inplace=True)
    segments = segments[["line", "inode", "jnode", "seg_seq", "loop", "boardings", "alightings", "volume"]].copy()

    return segments


def read_nwp_transit_vehicles(
    nwp_fp: Union[str, PathLike],
) -> pd.DataFrame:
    """A function to read the transit vehicles from a Network Package file (exported from Emme using the TMG Toolbox)
    into DataFrames.

    Args:
        nwp_fp (str | PathLike): File path to the network package.

    Returns:
        pd.DataFrame: DataFrame containing the transit vehicles.
    """
    nwp_fp = Path(nwp_fp)
    if not nwp_fp.exists():
        raise FileNotFoundError(f"File `{nwp_fp.as_posix()}` not found.")

    with zipfile.ZipFile(nwp_fp) as zf:
        # Get header
        header = None
        for i, line in enumerate(zf.open("vehicles.202"), start=1):
            line = line.strip().decode("utf-8")
            if line.startswith("c"):
                continue  # Skip comment lines
            if line.startswith("t vehicles"):
                header = i

        # Read data
        data_types = {
            "id": int,
            "description": str,
            "mode": str,
            "fleet_size": int,
            "seated_capacity": float,
            "total_capacity": float,
            "cost_time_coeff": float,
            "cost_distance_coeff": float,
            "energy_time_coeff": float,
            "energy_distance_coeff": float,
            "auto_equivalent": float,
        }
        vehicles = pd.read_csv(
            zf.open("vehicles.202"),
            index_col="id",
            usecols=data_types.keys(),
            dtype=data_types,
            skiprows=header,
            quotechar="'",
            delim_whitespace=True,
        )
        vehicles.index.name = "veh_id"

    return vehicles
