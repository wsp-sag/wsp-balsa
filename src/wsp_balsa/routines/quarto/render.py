# Based on https://github.com/quarto-dev/quarto-python/blob/main/quarto/render.py with modifications
# License under MIT License
# Copyright (c) 2020-2025 Posit Software, PBC

from __future__ import annotations

__all__ = [
    "render",
]

import os
import subprocess as sp
import tempfile

import yaml

from .quarto import find_quarto


def render(
    input,
    output_format=None,
    output_file=None,
    output_dir=None,
    execute=True,
    execute_params=None,
    execute_dir=None,
    cache=None,
    cache_refresh=False,
    kernel_keepalive=None,
    kernel_restart=False,
    debug=False,
    quiet=False,
    pandoc_args=None,
) -> None:
    """Interface to render a Quarto document from Python"""

    # params file to remove after render
    params_file = None

    # build args
    args = ["render", str(input)]

    if output_format is not None:
        args.extend(["--to", output_format])

    if output_file is not None:
        args.extend(["--output", str(output_file)])

    if output_dir is not None:
        args.extend(["--output-dir", str(output_dir)])

    if execute is not None:
        if execute is True:
            args.append("--execute")
        elif execute is False:
            args.append("--no-execute")

    if execute_params is not None:
        params_file = tempfile.NamedTemporaryFile(mode="w", prefix="quarto-params", suffix=".yml", delete=False)
        yaml.dump(execute_params, params_file)
        params_file.close()
        args.extend(["--execute-params", params_file.name])

    if execute_dir is not None:
        args.extend(["--execute-dir", str(execute_dir)])

    if cache is not None:
        if cache is True:
            args.append("--cache")
        elif cache is False:
            args.append("--no-cache")

    if cache_refresh is True:
        args.append("--cache-refresh")

    if kernel_keepalive is not None:
        args.extend(["--kernel-keepalive", str(kernel_keepalive)])

    if kernel_restart is True:
        args.append("--kernel-restart")

    if debug is True:
        args.append("--debug")

    if quiet is True:
        args.append("--quiet")

    # run process
    process = sp.Popen([str(find_quarto())] + args, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    try:
        while process.poll() is None:
            line = process.stderr.readline()  # Quarto writes progress info to stderr, so we read from there
            if not line.isspace() and line:
                print(line.rstrip())
        msg, err = process.communicate()
        if process.returncode:
            raise RuntimeError(err)
    finally:
        process.kill()
        if params_file is not None:
            os.remove(params_file.name)
