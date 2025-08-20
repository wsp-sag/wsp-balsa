from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def _parse_version(ver_str: str) -> Tuple[int, int, int]:
    return tuple([int(v) for v in ver_str.split(".")])


def read_calibration_target_table(model_package_dict: Dict[str, Any]) -> pd.DataFrame:
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


def read_utility_expression_table(choice_component_spec: dict) -> pd.DataFrame:
    attribute_info = pd.DataFrame(choice_component_spec["utility_expression_table"]["attribute_info"]).set_index("name")
    component_data = [el.split(";") for el in choice_component_spec["utility_expression_table"]["data"]]
    df = pd.DataFrame.from_records(component_data, columns=attribute_info.index.tolist())
    if not df.empty:
        if choice_component_spec["utility_specification_type"] == "wide":
            df.set_index(["description", "agent_filter", "agent_expression"], inplace=True)
            df.columns.name = "alternative_filter"
            df = df.stack().to_frame("coefficient").reset_index()
            df["alternative_filter"] = df["alternative_filter"].map(attribute_info["description"])
            df["alternative_expression"] = ""
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
