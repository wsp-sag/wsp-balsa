import logging
import sys
import traceback as tb
from contextlib import contextmanager
from enum import Enum
from json import dumps as json_to_str
from json import loads as parse_json
from logging import FileHandler, Formatter, Logger, LogRecord
from os import PathLike
from typing import Any, Dict, Generator, Union

try:
    from inro.modeller import logbook_write

    EMME_ENV = True
except (ImportError, RuntimeError):
    logbook_write = None
    EMME_ENV = False


# region Constants

_FMT_STRING = "%(asctime)s %(levelname)s %(name)s {arrow} %(message)s"
_UNC_ARROW = "➔"
_ASCII_ARROW = "->"
_TIP_LEVEL = 21  # Important log info
_REPORT_LEVEL = 19  # Model results
_SUBPROC_LEVEL = 14
_SUBPROC_ERR_LEVEL = 41
_FANCY_FORMAT = "FANCY"
_BASIC_FORMAT = "BASIC"
_JSON_FORMAT = "JSON"


class LogFormats(Enum):
    FANCY = _FANCY_FORMAT
    BASIC = _BASIC_FORMAT
    JSON = _JSON_FORMAT


# endregion


# region Filter Classes


class _RangeFilter(object):

    def __init__(self, low: int, high: int):
        self._low = int(low)
        self._high = int(high)

    def filter(self, record: LogRecord) -> int:
        return int(self._low <= record.levelno <= self._high)


# endregion


# region Formatter Classes


class _SwitchFormatter(Formatter):

    def __init__(self, default_format: Union[Formatter, str], level_formats: Dict[int, Union[Formatter, str]]):
        super(_SwitchFormatter, self).__init__()

        def make_formatter(item: Union[Formatter, str]) -> Formatter:
            return Formatter(item) if isinstance(item, str) else item

        self._default = make_formatter(default_format)
        self._formats = {lvl: make_formatter(f) for lvl, f in level_formats.items()}

    def format(self, record: LogRecord) -> str:
        level = record.levelno
        if level in self._formats:
            return self._formats[level].format(record)
        return self._default.format(record)


class _JsonFormatter(Formatter):

    def format(self, record: LogRecord) -> str:
        keys = ["levelname", "name", "msg", "created", "levelno"]
        to_json = {key: getattr(record, key) for key in keys}
        to_json["asctime"] = self.formatTime(record)
        return json_to_str(to_json)


# endregion


class ModelLogger(Logger):

    def __init__(self, name: str, level: int = logging.NOTSET):
        """ModelLogger extends the standard Python Logger, adding additional statements such as ``.report()``."""
        super(ModelLogger, self).__init__(name, level)
        self._all_load_failures = []

    def report(self, msg, *args, **kwargs):
        """Report useful model statistics or results to the user. Distinct from ``.info()`` which provides status
        information. Printed in green when colours are available."""
        self.log(_REPORT_LEVEL, msg, *args, **kwargs)

    def tip(self, msg, *args, **kwargs):
        """Provide a more significant status statement (e.g. new section of the model). Similar to ``.info()``, but
        more emphasized. Printed in blue when colours are available."""
        self.log(_TIP_LEVEL, msg, *args, **kwargs)

    def subproc_message(self, msg, *args, **kwargs):
        """Report subprocess messages to the user. Distinct from ``.info()`` which provides status information."""
        self.log(_SUBPROC_LEVEL, msg, *args, **kwargs)

    def subproc_err(self, msg, *args, **kwargs):
        """Report subprocess errors to the user. Distinct from ``.error()`` which provides error information."""
        self.log(_SUBPROC_ERR_LEVEL, msg, *args, **kwargs)

    def log_json(self, json_string: str):
        json_dict: Dict[str, Any] = parse_json(json_string)
        record = logging.makeLogRecord(json_dict)
        self.handle(record)

    def load_failure(self, msg: str):
        self._all_load_failures.append(msg)
        self.warning(msg)

    if EMME_ENV:

        def flush_load_failures(self):
            body = "\n".join(self._all_load_failures)
            logbook_write(name="List of all errors in model configuration", value=body)

        def __del__(self):
            self.flush_load_failures()


def _prep_fancy_formatter():
    raw_fmt = _FMT_STRING.format(arrow=_UNC_ARROW)
    fmt_string = str("".join(["\x1b[{colour}m", raw_fmt, "\x1b[0m"]))

    debug_formatter = logging.Formatter(fmt_string.format(colour=37))  # Grey colour
    subproc_formatter = logging.Formatter(fmt_string.format(colour=37))  # Grey colour
    report_formatter = logging.Formatter(fmt_string.format(colour=32))  # Green colour
    info_formatter = logging.Formatter(fmt_string.format(colour=0))  # Default colour
    tip_formatter = logging.Formatter(fmt_string.format(colour=34))  # Blue colour
    warn_formatter = logging.Formatter(fmt_string.format(colour=31))  # Red colour
    error_formatter = logging.Formatter(fmt_string.format(colour=41))  # Red BG colour
    subproc_err_formatter = logging.Formatter(fmt_string.format(colour=41))  # Red BG colour
    critical_formatter = logging.Formatter(fmt_string.format(colour="1m\x1b[41"))  # Bold on red BG

    switch_formatter = _SwitchFormatter(
        raw_fmt,
        {
            logging.INFO: info_formatter,
            logging.WARNING: warn_formatter,
            _TIP_LEVEL: tip_formatter,
            _REPORT_LEVEL: report_formatter,
            logging.DEBUG: debug_formatter,
            logging.ERROR: error_formatter,
            logging.CRITICAL: critical_formatter,
            _SUBPROC_LEVEL: subproc_formatter,
            _SUBPROC_ERR_LEVEL: subproc_err_formatter,
        },
    )

    return switch_formatter


# region Helper Functions


def init_root(
    name: str,
    *,
    stream_format: Union[str, LogFormats] = LogFormats.FANCY,
    log_debug: bool = True,
) -> ModelLogger:
    """Initialize a ModelLogger

    Args:
        root_name (str): The logger's name to use
        stream_format (str | LogFormats, optional): Defaults to ``LogFormats.FANCY``. Logging format options to use.
            Options include ``BASIC``, ``FANCY``, or ``JSON``
        log_debug (bool, optional): Defaults to ``True``. A flag to log debug-level messages
    """

    logging.addLevelName(_SUBPROC_ERR_LEVEL, "SUBPROC_ERR")
    logging.addLevelName(_TIP_LEVEL, "TIP")
    logging.addLevelName(_REPORT_LEVEL, "REPORT")
    logging.addLevelName(_SUBPROC_LEVEL, "SUBPROC")
    logging.setLoggerClass(ModelLogger)

    root_logger: ModelLogger = logging.getLogger(name)  # Will return a ModelLogger based on the previous line
    root_logger.propagate = True
    root_logger.setLevel(1 if log_debug else _REPORT_LEVEL)

    stream_format = LogFormats(stream_format)
    if stream_format == LogFormats.FANCY:
        stdout_formatter = _prep_fancy_formatter()
    elif stream_format == LogFormats.BASIC:
        stdout_formatter = logging.Formatter(_FMT_STRING.format(arrow=_ASCII_ARROW))
    elif stream_format == LogFormats.JSON:
        stdout_formatter = _JsonFormatter()
    else:
        stdout_formatter = logging.Formatter(stream_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_RangeFilter(0, 100).filter)
    stdout_handler.setFormatter(stdout_formatter)

    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)  # Remove any existing handlers
    root_logger.addHandler(stdout_handler)

    return root_logger


def get_model_logger(name: str) -> ModelLogger:
    """Retrieve a ModelLogger with the specified name"""
    logging.setLoggerClass(ModelLogger)
    return logging.getLogger(name)  # Will return a ModelLogger based on the previous line


@contextmanager
def log_to_file(
    file_name: Union[str, PathLike],
    name: str,
    *,
    append: bool = False,
    raw_ascii: bool = False,
) -> Generator:
    """Context manager for opening and closing a logfile. Cleans up its file handler on exit.

    This is especially important during batch runs, because loggers are module-based (e.g. global). Without the cleanup,
    old file handlers would stick around and get written to.

    Args:
        file_name (str | PathLike): The filepath of the log file to write to
        name (str): The name of the logger to write log records from
        append (bool, optional): Defaults to ``False``. Option to append new log records to an existing log file
        raw_ascii (bool, optional): Defaults to ``False``. Ensures log file only contains valid ASCII characters
    """
    root = logging.getLogger(name)

    write_mode = "a" if append else "w"
    handler = FileHandler(file_name, mode=write_mode, encoding="utf-8")

    arrow = _ASCII_ARROW if raw_ascii else _UNC_ARROW
    fmt_str = _FMT_STRING.format(arrow=arrow)
    handler.setFormatter(Formatter(fmt_str))

    handler.addFilter(_RangeFilter(0, 100).filter)

    root.addHandler(handler)
    try:
        yield
    except:
        with open(file_name, mode="a") as writer:
            writer.write(str("\n" + "-" * 100 + "\n\n"))
            writer.write(tb.format_exc())
        raise
    finally:
        root.removeHandler(handler)


# endregion
