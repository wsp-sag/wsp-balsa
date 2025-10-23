from __future__ import annotations

__all__ = [
    "read_feat",
    "read_table_calculator_result_attributes",
    "read_choice_model_component_tables",
    "read_choice_model_calibration_automata_table",
    "read_utility_expression_table",
    "read_calibration_target_table",
]

from json import loads
from os import PathLike
from typing import Any, Dict, Tuple, Union

import pandas as pd

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

# region Model Package


def _parse_version(ver_str: str) -> Tuple[int, int, int]:
    return tuple([int(v) for v in ver_str.split(".")])


def read_calibration_target_table(model_package_dict: Dict[str, Any]) -> pd.DataFrame:
    """Reads the calibration target table from an AGENT model package specification

    Args:
        model_package_dict (dict): The AGENT model package to parse, as a Python dictionary

    Returns:
        pd.DataFrame
    """
    if _parse_version(model_package_dict["version"]) >= _parse_version("1.3.16"):
        spec = model_package_dict["calibration_targets_table"]
        attribute_info = pd.DataFrame(spec["attribute_info"]).set_index("name")
        component_data = [el.split(";") for el in spec["data"]]
        df = pd.DataFrame.from_records(component_data, columns=attribute_info.index.tolist())
    else:
        df = pd.DataFrame.from_records(model_package_dict["calibration_targets"])
    df["target_min_value"] = pd.to_numeric(df["target_min_value"])
    df["target_max_value"] = pd.to_numeric(df["target_max_value"])
    df["importance"] = pd.to_numeric(df["importance"])
    df["min_calibrated_value"] = pd.to_numeric(df["min_calibrated_value"])
    df["max_calibrated_value"] = pd.to_numeric(df["max_calibrated_value"])
    df["value"] = pd.to_numeric(df["value"])

    return df.set_index("name")


def read_choice_model_component_tables(model_step_dict: Dict[str, Any]) -> pd.DataFrame:
    """Reads the choice components from an AGENT choice model step

    Args:
        model_step_dict (dict): The choice component model step to read in an AGENT model package spec, as a dictionary

    Returns:
        pd.DataFrame
    """
    if model_step_dict["procedure_type"] != "CHOICE_MODEL":
        raise ValueError(f"Model step `{model_step_dict['name']}` is not a choice model")
    choice_components = []
    for i, spec in enumerate(model_step_dict["choice_components"]):
        df = read_utility_expression_table(spec)
        df.insert(0, "choice_component_id", i)
        df.insert(1, "choice_component_name", spec["name"])
        choice_components.append(df)
    choice_components: pd.DataFrame = pd.concat(choice_components, axis=0, ignore_index=True)

    return choice_components


def read_choice_model_calibration_automata_table(model_step_dict: Dict[str, Any]) -> pd.DataFrame:
    """Reads the calibration instructions table from an AGENT choice model step

    Args:
        model_step_dict (dict): The choice component model step to read in an AGENT model package spec, as a dictionary

    Returns:
        pd.DataFrame
    """
    if model_step_dict["procedure_type"] != "CHOICE_MODEL":
        raise ValueError(f"Model step `{model_step_dict['name']}` is not a choice model")
    spec = model_step_dict["calibration_automata_table"]
    attribute_info = pd.DataFrame(spec["attribute_info"]).set_index("name")
    component_data = [el.split(";") for el in spec["data"]]
    df = pd.DataFrame.from_records(component_data, columns=attribute_info.index.tolist())
    df["value"] = pd.to_numeric(df["value"])
    df["min_calibrated_value"] = pd.to_numeric(df["min_calibrated_value"])
    df["max_calibrated_value"] = pd.to_numeric(df["max_calibrated_value"])

    return df


def read_table_calculator_result_attributes(model_step_dict: Dict[str, Any]) -> pd.DataFrame:
    """Reads the result attributes from an AGENT table calculator model step

    Args:
        model_step_dict (dict): The table calculator model step to read in an AGENT model package spec, as a dictionary

    Returns:
        pd.DataFrame
    """
    if model_step_dict["procedure_type"] != "TABLE_CALCULATOR":
        raise ValueError(f"Model step `{model_step_dict['name']}` is not a table calculator")
    df = pd.DataFrame(model_step_dict["result_attributes"])

    return df


def read_utility_expression_table(choice_component_dict: dict) -> pd.DataFrame:
    """Reads the utility expression table from an AGENT model step choice component

    Args:
        choice_component_dict (dict): The choice component of a model step in an AGENT model package spec, as a
            dictionary

    Returns:
        pd.DataFrame
    """
    attribute_info = pd.DataFrame(choice_component_dict["utility_expression_table"]["attribute_info"]).set_index("name")
    component_data = [el.split(";") for el in choice_component_dict["utility_expression_table"]["data"]]
    df = pd.DataFrame.from_records(component_data, columns=attribute_info.index.tolist())
    if not df.empty:
        # Handle wide format
        if choice_component_dict["utility_specification_type"] == "wide":
            df.set_index(["description", "agent_filter", "agent_expression"], inplace=True)
            df.columns.name = "alternative_filter"
            df = df.stack().to_frame("coefficient").reset_index()
            df["alternative_filter"] = df["alternative_filter"].map(attribute_info["description"])
            df["alternative_expression"] = ""

        # Finalize
        if "agent_filter" not in df:
            df["agent_filter"] = ""
        if "agent_expression" not in df:
            df["agent_expression"] = ""
        df = df[
            [
                "description",
                "alternative_filter",
                "alternative_expression",
                "agent_filter",
                "agent_expression",
                "coefficient",
            ]
        ].copy()

    return df


# endregion
