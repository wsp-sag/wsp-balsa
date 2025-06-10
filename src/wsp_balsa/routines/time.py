from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict


def generate_time_bin_labels(
    day_start_minute: int,
    day_end_minute: int,
    *,
    time_resolution_minute: int = 30,
    base_year: int = None,
    base_month: int = None,
    base_day: int = None,
) -> Dict[int, datetime]:
    """Generate time bin labels based on start time, end time, and time resolution.

    Args:
        day_start_minute (int): Start time of day, in minutes past midnight
        day_end_minute (int): End time of day, in minutes past midnight
        time_resolution_minute (int, optional): Defaults to ``30``. The number of minutes in each time bin
        base_year (int, optional): Defaults to ``None``. The year to use as a base. If ``None``, the current year will
            be used.
        base_month (int, optional): Defaults to ``None``. The month to use as a base. If ``None``, the current month
            will be used.
        base_day (int, optional): Defaults to ``None``. The day to use as a base. If ``None``, the current day will
            be used.

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

    base_dt = datetime.today()
    if base_year is not None:
        base_dt = base_dt.replace(year=base_year)
    if base_month is not None:
        base_dt = base_dt.replace(month=base_month)
    if base_day is not None:
        base_dt = base_dt.replace(day=base_day)

    start_time = base_dt.replace(hour=start_hr, minute=start_min, second=0, microsecond=0)
    labels = {(i + 1): start_time + timedelta(seconds=time_resolution_minute * 60 * i) for i in range(n_intervals)}

    return labels
