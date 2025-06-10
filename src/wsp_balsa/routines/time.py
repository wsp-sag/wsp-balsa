from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict


def generate_time_bin_labels(
    day_start_minute: int,
    day_end_minute: int,
    *,
    time_resolution_minute: int = 30,
    base_dt: datetime = None,
) -> Dict[int, datetime]:
    """Generate time bin labels based on start time, end time, and time resolution.

    Args:
        day_start_minute (int): Start time of day, in minutes past midnight
        day_end_minute (int): End time of day, in minutes past midnight
        time_resolution_minute (int, optional): Defaults to ``30``. The number of minutes in each time bin
        base_dt (datetime, optional): Defaults to ``None``. A datetime object to use as a base for generating the time
            bin labels. All properties of the provided datetime object will be inherited except hour, minute, second,
            and microsecond. If ``None``, the current date will be used.

    Returns:
        dict[int, datetime]: A dictionary with the time bins as its keys and datetime objects as its values
    """

    if day_start_minute > day_end_minute:
        raise ValueError("`day_end_minute` must be > `day_start_minute`")

    diff = day_end_minute - day_start_minute
    if diff < time_resolution_minute:
        raise ValueError("`day_start_minute - day_end_minute` must be > `time_resolution_minute`")

    start_hr, start_min = divmod(day_start_minute, 60)
    n_intervals = diff // time_resolution_minute

    if base_dt is None:
        base_dt = datetime.today()
    start_time = base_dt.replace(hour=start_hr, minute=start_min, second=0, microsecond=0)
    labels = {(i + 1): start_time + timedelta(seconds=time_resolution_minute * 60 * i) for i in range(n_intervals)}

    return labels
