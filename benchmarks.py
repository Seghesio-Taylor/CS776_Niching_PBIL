"""Benchmark functions and encoding utilities for GA and PBIL algorithms.

This module defines the multimodal benchmark functions from Deb & Goldberg (1989)
along with helper functions to map between binary chromosomes and real-valued
phenotypes. Phenotypic distance functions are also provided for use in
restricted mating and niche detection.

Functions:
    bits_to_real_1d(bits): Convert a 30-bit boolean array to a real number in [0,1].
    bits_to_real_2d(bits): Convert a 30-bit boolean array to a pair of real numbers
        in [-6,6] × [-6,6]. The first 15 bits encode x1 and the next 15 bits encode x2.
    real_to_bits_1d(x): Convert a real number in [0,1] to a 30‑bit boolean array.
    f1(x): Evaluate the first benchmark function at x.
    f2(x): Evaluate the second benchmark function at x.
    f3(x1, x2): Evaluate the third benchmark function (Himmelblau) at (x1, x2).
    distance_1d(xi, xj): Compute absolute distance between two 1D phenotypes.
    distance_2d(xi, xj): Compute squared Euclidean distance between two 2D phenotypes.

All fitness functions are defined as maximization problems. For f3 the value is
the negative of the classical Himmelblau function so that maxima correspond to
the known minima of the original function.

"""

from __future__ import annotations

import numpy as np
import math
from typing import Iterable, Tuple, List


# Constants for encoding
_BITS_TOTAL = 30
_BITS_HALF = _BITS_TOTAL // 2
_MAX_INT_30 = (1 << _BITS_TOTAL) - 1
_MAX_INT_15 = (1 << _BITS_HALF) - 1


def bits_to_int(bits: np.ndarray) -> int:
    """Convert a 1D array of 0/1 bits to an integer.

    Parameters
    ----------
    bits : np.ndarray
        Array of shape (n,) containing 0/1 integers or booleans. The most
        significant bit should be at index 0.

    Returns
    -------
    int
        Integer representation of the bit string.
    """
    # Interpret bits as a binary number. Use bit shifts for speed.
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def bits_to_real_1d(bits: np.ndarray) -> float:
    """Decode a 30-bit chromosome into a real value in [0, 1].

    The chromosome is interpreted as an integer b in the range [0, 2^30 - 1],
    then mapped linearly to x = b / (2^30 - 1).

    Parameters
    ----------
    bits : np.ndarray
        Boolean or 0/1 array of length 30.

    Returns
    -------
    float
        Real value x in [0,1].
    """
    if bits.shape[0] != _BITS_TOTAL:
        raise ValueError(f"Expected 30 bits for 1D encoding, got {bits.shape[0]}")
    b = bits_to_int(bits)
    return b / _MAX_INT_30


def bits_to_real_2d(bits: np.ndarray) -> Tuple[float, float]:
    """Decode a 30-bit chromosome into a pair (x1, x2) in [-6, 6]^2.

    The first 15 bits encode b1; the second 15 bits encode b2. Each value is
    mapped linearly to the interval [-6, 6] via x = -6 + 12 * b / (2^15 - 1).

    Parameters
    ----------
    bits : np.ndarray
        Boolean or 0/1 array of length 30.

    Returns
    -------
    Tuple[float, float]
        (x1, x2) in [-6,6]^2.
    """
    if bits.shape[0] != _BITS_TOTAL:
        raise ValueError(f"Expected 30 bits for 2D encoding, got {bits.shape[0]}")
    b1 = bits_to_int(bits[:_BITS_HALF])
    b2 = bits_to_int(bits[_BITS_HALF:])
    x1 = -6.0 + 12.0 * b1 / _MAX_INT_15
    x2 = -6.0 + 12.0 * b2 / _MAX_INT_15
    return x1, x2


def real_to_bits_1d(x: float) -> np.ndarray:
    """Encode a real value x in [0,1] into a 30-bit chromosome.

    Values outside [0,1] are clipped. The returned array is of dtype int8.

    Parameters
    ----------
    x : float
        Real number in [0,1].

    Returns
    -------
    np.ndarray
        Array of shape (30,) with values 0 or 1.
    """
    x_clamped = max(0.0, min(1.0, float(x)))
    b = int(round(x_clamped * _MAX_INT_30))
    bits = np.zeros(_BITS_TOTAL, dtype=np.int8)
    for i in range(_BITS_TOTAL - 1, -1, -1):
        bits[i] = b & 1
        b >>= 1
    return bits


def f1(x: float) -> float:
    """First benchmark function (five equal peaks).

    f1(x) = sin^6(5πx). The domain is [0,1] and the function is maximized.

    Parameters
    ----------
    x : float
        Input value in [0,1].

    Returns
    -------
    float
        Function value.
    """
    # Use math.sin for speed; exponentiate after computing sine.
    # Multiply by pi and 5 first.
    s = math.sin(5 * math.pi * x)
    return s * s * s * s * s * s  # s**6


def f2(x: float) -> float:
    """Second benchmark function (five unequal peaks).

    f2(x) = exp(-2 ln(2) * ((x - 0.1)/0.8)^2) * sin^6(5πx).
    The domain is [0,1] and the function is maximized.

    Parameters
    ----------
    x : float
        Input value in [0,1].

    Returns
    -------
    float
        Function value.
    """
    s = math.sin(5 * math.pi * x)
    term = (x - 0.1) / 0.8
    # -2 ln 2 * term^2 = -(2*ln2) * term^2
    envelope = math.exp(-2.0 * math.log(2.0) * term * term)
    return envelope * (s * s * s * s * s * s)


def f3(x1: float, x2: float) -> float:
    """Third benchmark function: negative Himmelblau function.

    Original Himmelblau function h(x1,x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2.
    For maximization we return -h(x1,x2).

    Parameters
    ----------
    x1 : float
        First coordinate in [-6,6].
    x2 : float
        Second coordinate in [-6,6].

    Returns
    -------
    float
        Negated Himmelblau function value.
    """
    a = x1 * x1 + x2 - 11.0
    b = x1 + x2 * x2 - 7.0
    return -(a * a + b * b)


def distance_1d(xi: float, xj: float) -> float:
    """Phenotypic distance between two 1D values.

    Defined as the absolute difference |xi - xj|.
    """
    return abs(xi - xj)


def distance_2d(xi: Tuple[float, float], xj: Tuple[float, float]) -> float:
    """Phenotypic distance between two 2D values.

    Defined as squared Euclidean distance: (x1_i - x1_j)^2 + (x2_i - x2_j)^2.
    This squared distance is used in Deb & Goldberg for mating restriction and
    niche detection. Using the squared distance avoids an expensive square root.
    """
    dx = xi[0] - xj[0]
    dy = xi[1] - xj[1]
    return dx * dx + dy * dy


def get_function(func_id: str):
    """Return the appropriate function handle given its identifier.

    Parameters
    ----------
    func_id : str
        Identifier: 'F1', 'F2', or 'F3'. Case-insensitive.

    Returns
    -------
    callable
        Function handle that accepts either x (for F1/F2) or (x1,x2) for F3.
    """
    fid = func_id.upper()
    if fid == 'F1':
        return f1
    elif fid == 'F2':
        return f2
    elif fid == 'F3':
        return f3
    else:
        raise ValueError(f"Unknown function id: {func_id}")


# Known peak positions for F3 (Himmelblau minima) for analysis
HIMMELBLAU_MINIMA = [
    (3.0, 2.0),
    (-2.805118, 3.131312),
    (-3.779310, -3.283186),
    (3.584428, -1.848126),
]
