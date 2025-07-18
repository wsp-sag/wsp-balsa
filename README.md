# Balsa (wsp-balsa)

![PyPI - Version](https://img.shields.io/pypi/v/wsp-balsa)

Balsa is a collection of functions and tools for Python to facilitate travel demand forecasting applications and analyses. It is designed to work the the “scientific stack” of Python, namely NumPy, Pandas, and Matplotlib; which are optimized for speed and usability. Most of balsa consists of standalone functions - for input/output, for analysis, etc. - as well as a few lightweight class-based data structures for specific applications.

> [!IMPORTANT]
> As of v2.0, this package is imported using `wsp_balsa` instead of `balsa`

## Key features

- I/O routines to convert from binary matrix formats (INRO, OMX, and more) to Pandas DataFrames and Series
- Matrix operations such as balancing, dis/aggregation, triple-indexing, and bucket rounding
- Plotting functions such a Trip Length Frequency Distributions
- Pretty Logging utilities for use in program applications
- and more!

Balsa is compatible with Python 3.7+

## Installation

> [!NOTE]
> **For TRESO v1.4 (and older) users:** TRESO is only compatible with the [`v0.6.1`](https://github.com/wsp-sag/balsa/releases/tag/v0.6.1) release of Balsa, which can only be installed directly from GitHub using `pip`.

Balsa can be installed with the following command:

```batch
pip install wsp-balsa
```

### With `pip` directly from GitHub

Balsa can be installed directly from GitHub using `pip` by running the following command:

```batch
pip install git+https://github.com/wsp-sag/balsa.git
```
