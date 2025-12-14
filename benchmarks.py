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
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def bits_to_real_1d(bits: np.ndarray) -> float:
    if bits.shape[0] != _BITS_TOTAL:
        raise ValueError(f"Expected 30 bits for 1D encoding, got {bits.shape[0]}")
    b = bits_to_int(bits)
    return b / _MAX_INT_30

def bits_to_real_2d(bits: np.ndarray) -> Tuple[float, float]:
    if bits.shape[0] != _BITS_TOTAL:
        raise ValueError(f"Expected 30 bits for 2D encoding, got {bits.shape[0]}")
    b1 = bits_to_int(bits[:_BITS_HALF])
    b2 = bits_to_int(bits[_BITS_HALF:])
    x1 = -6.0 + 12.0 * b1 / _MAX_INT_15
    x2 = -6.0 + 12.0 * b2 / _MAX_INT_15
    return x1, x2

def real_to_bits_1d(x: float) -> np.ndarray:
    x_clamped = max(0.0, min(1.0, float(x)))
    b = int(round(x_clamped * _MAX_INT_30))
    bits = np.zeros(_BITS_TOTAL, dtype=np.int8)
    for i in range(_BITS_TOTAL - 1, -1, -1):
        bits[i] = b & 1
        b >>= 1
    return bits

def real_to_bits_2d(x1: float, x2: float) -> np.ndarray:
    x1c = max(-6.0, min(6.0, float(x1)))
    x2c = max(-6.0, min(6.0, float(x2)))
    b1 = int(round((x1c + 6.0) * _MAX_INT_15 / 12.0))
    b2 = int(round((x2c + 6.0) * _MAX_INT_15 / 12.0))
    bits = np.zeros(_BITS_TOTAL, dtype=np.int8)
    for i in range(_BITS_HALF - 1, -1, -1):
        bits[i] = b1 & 1
        b1 >>= 1
    for i in range(_BITS_TOTAL - 1, _BITS_HALF - 1, -1):
        bits[i] = b2 & 1
        b2 >>= 1
    return bits

def f1(x: float) -> float:
    s = math.sin(5 * math.pi * x)
    return s * s * s * s * s * s  # s**6

def f2(x: float) -> float:
    s = math.sin(5 * math.pi * x)
    term = (x - 0.1) / 0.8
    envelope = math.exp(-2.0 * math.log(2.0) * term * term)
    return envelope * (s * s * s * s * s * s)

def f3(x1: float, x2: float) -> float:
    a = x1 * x1 + x2 - 11.0
    b = x1 + x2 * x2 - 7.0
    return -(a * a + b * b)

def distance_1d(xi: float, xj: float) -> float:
    return abs(xi - xj)

def distance_2d(xi: Tuple[float, float], xj: Tuple[float, float]) -> float:
    dx = xi[0] - xj[0]
    dy = xi[1] - xj[1]
    return dx * dx + dy * dy

def get_function(func_id: str):
    fid = func_id.upper()
    if fid == 'F1':
        return f1
    elif fid == 'F2':
        return f2
    elif fid == 'F3':
        return f3
    else:
        raise ValueError(f"Unknown function id: {func_id}")
