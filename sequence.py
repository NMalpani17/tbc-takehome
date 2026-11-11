# (1)
# Complete the sequence_calculator function, which should
# Return the n-th number of the sequence S_n, defined as:
# S_n = 3*S_(n-1) - S_(n-2), with S_0 = 0 and S_1 = 1.
# Your implementation should minimize the execution time.
#
# (2)
# Find the time complexity of the proposed solution, using
# the "Big O" notation, and explain in detail why such
# complexity is obtained, for n ranging from 0 to at least
# 100000. HINT: you are dealing with very large numbers!
#
# (3)
# Plot the execution time VS n (again, for n ranging from 0
# to at least 100000).
#
# (4)
# Confirm that the empirically obtained time complexity curve
# from (3) matches the claimed time complexity from (2) (e.g.
# by using curve fitting techniques).

"""Fast computation of S_n = 3*S_(n-1) - S_(n-2), S_0 = 0, S_1 = 1.

Key identity: S_n = F(2n)
------------------------
The sequence 0, 1, 3, 8, 21, 55, ... is every other Fibonacci number
(F0, F2, F4, F6, F8, F10, ...). Proof: using F(m) = F(m-1) + F(m-2) twice,

    F(2n) = F(2n-1) + F(2n-2)
          = (F(2n-2) + F(2n-3)) + F(2n-2)
          = 2F(2n-2) + (F(2n-2) - F(2n-4))      since F(2n-3) = F(2n-2) - F(2n-4)
          = 3F(2n-2) - F(2n-4).

So G(n) = F(2n) obeys the same recurrence as S_n, with the same initial
values G(0) = F(0) = 0 and G(1) = F(2) = 1. By induction, S_n = F(2n).

Algorithm: fast-doubling Fibonacci
----------------------------------
    F(2k)   = F(k) * (2F(k+1) - F(k))
    F(2k+1) = F(k)^2 + F(k+1)^2

Walking the bits of n from the most significant one, each iteration turns
(F(k), F(k+1)) into (F(2k), F(2k+1)), then shifts one place on if the bit
is 1. After bit_length(n) iterations we hold (F(n), F(n+1)), and one more
doubling formula gives F(2n) = S_n.

Time complexity: O(n^log2(3)) = O(n^1.585), not O(log n)
-------------------------------------------------------
The loop runs only O(log n) times, but each iteration does arithmetic on
huge integers, so counting iterations is misleading:

* Size of the numbers. S_n = F(2n) ~ phi^(2n) / sqrt(5), so S_n has about
  2 * log2(phi) * n ~ 1.39n bits (~0.42n decimal digits; S_100000 has
  ~41,800 digits). The operands of the final multiplication, F(n) and
  F(n+1), have ~0.69n bits.
* Cost of one multiplication. CPython multiplies large ints with
  Karatsuba, which costs M(b) = O(b^log2(3)) = O(b^1.585) for b-bit
  operands. Additions, subtractions and shifts are only O(b).
* Summing over iterations. Operand size doubles every iteration, so going
  back one iteration halves b and divides the Karatsuba cost by
  2^1.585 = 3. The costs form a geometric series with ratio 1/3, dominated
  by the last multiplications:
      M(b) * (1 + 1/3 + 1/9 + ...) = 1.5 * M(b),   b ~ 0.69n,
  so the total is O(n^1.585). The O(log n) factor from the number of
  iterations is absorbed by the geometric series.

For comparison, the naive recurrence loop does n steps, each an O(b)
addition on numbers of up to ~1.39n bits, so it is O(n^2).

Measured exponents differ slightly from the asymptotic ones. CPython only
uses Karatsuba above ~2100-bit operands and falls back to O(b^2)
schoolbook multiplication for the pieces below that, so at these sizes
the recursion is only a few levels deep and the effective exponent sits a
little above 1.585. The naive loop also pays a fixed interpreter cost per
iteration, an O(n) term that pulls its fitted exponent a little below 2.

The benchmark times only the computation of S_n, not its conversion to a
decimal string, which is a separate cost and not part of computing S_n.

Run with: python sequence.py
Plots are saved to plots/sequence_time.png and plots/sequence_loglog.png.
"""

import math
import sys
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TARGET_N = 100_000
EDGE_DIGITS = 20
VERIFY_MAX_N = 200

BENCHMARK_MAX_N = 100_000
# 41 points gives a round step of 2500.
BENCHMARK_NUM_POINTS = 41
FAST_REPEATS = 7
FAST_CALLS_PER_REPEAT = 5
# The naive loop takes a large fraction of a second per call near n = 100000,
# so it gets fewer repeats to keep the whole script well under a minute.
NAIVE_REPEATS = 2
NAIVE_CALLS_PER_REPEAT = 1

# Below this, the timings are dominated by interpreter overhead, not big-int
# arithmetic, so they would bias the exponent fit.
FIT_MIN_N = 5_000
INITIAL_EXPONENT_GUESS = 1.5
KARATSUBA_EXPONENT = math.log2(3)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FIGURE_DPI = 150
MS_PER_S = 1e3
LINE_WIDTH = 2.0
MARKER_SIZE = 5
GRID_ALPHA = 0.3
FAST_COLOR = "#2a78d6"
NAIVE_COLOR = "#eb6834"
FIT_PLOT_POINTS = 200


@dataclass(frozen=True)
class PowerLawFit:
    """Fit of t = scale_s * (n / BENCHMARK_MAX_N) ** exponent."""

    scale_s: float
    exponent: float
    exponent_std: float


def validate_index(n: int) -> None:
    """Raise ValueError unless n is a non-negative int."""
    # bool is a subclass of int, but True/False are not meaningful indices.
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"n must be an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")


def sequence_calculator(n: int) -> int:
    """Return S_n = F(2n) by fast-doubling Fibonacci; O(n^1.585) time."""
    validate_index(n)
    fib_k, fib_k_plus_1 = 0, 1  # F(0), F(1)
    for bit in format(n, "b"):
        fib_2k = fib_k * (2 * fib_k_plus_1 - fib_k)
        fib_2k_plus_1 = fib_k * fib_k + fib_k_plus_1 * fib_k_plus_1
        if bit == "1":
            fib_k, fib_k_plus_1 = fib_2k_plus_1, fib_2k + fib_2k_plus_1
        else:
            fib_k, fib_k_plus_1 = fib_2k, fib_2k_plus_1
    # Now fib_k = F(n) and fib_k_plus_1 = F(n+1); one more doubling gives F(2n).
    return fib_k * (2 * fib_k_plus_1 - fib_k)


def naive_sequence(n: int) -> int:
    """Return S_n by iterating the recurrence directly; O(n^2) time."""
    validate_index(n)
    previous, current = 0, 1  # S_0, S_1
    for _ in range(n):
        previous, current = current, 3 * current - previous
    return previous


def verify_against_naive(max_n: int) -> None:
    """Raise AssertionError if the two implementations differ for any n < max_n."""
    for n in range(max_n):
        fast, naive = sequence_calculator(n), naive_sequence(n)
        if fast != naive:
            raise AssertionError(f"Mismatch at n = {n}: {fast} != {naive}")


def summarize_digits(value: int, edge_digits: int) -> tuple[int, str, str]:
    """Return (digit count, leading digits, trailing digits) of a non-negative int."""
    # CPython refuses int -> str beyond 4300 digits by default (a DoS guard).
    # Lift the limit only for this conversion and restore it afterwards.
    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        digits = str(value)
    finally:
        sys.set_int_max_str_digits(previous_limit)
    return len(digits), digits[:edge_digits], digits[-edge_digits:]


def benchmark(
    func: Callable[[int], int],
    n_values: IntArray,
    repeats: int,
    calls_per_repeat: int,
) -> FloatArray:
    """Return the time per call [s] of func(n) for each n.

    Takes the minimum over repeats, because noise from the OS and other
    processes only ever adds time.
    """
    seconds_per_call = np.empty(len(n_values))
    for i, n in enumerate(n_values):
        n_int = int(n)
        totals = timeit.repeat(
            lambda: func(n_int), number=calls_per_repeat, repeat=repeats
        )
        seconds_per_call[i] = min(totals) / calls_per_repeat
    return seconds_per_call


def power_law(n: FloatArray, scale_s: float, exponent: float) -> FloatArray:
    """t = scale_s * (n / BENCHMARK_MAX_N) ** exponent.

    Dividing n by BENCHMARK_MAX_N makes scale_s the time at the largest n,
    which keeps both parameters of order one and the fit well conditioned.
    """
    return scale_s * (n / BENCHMARK_MAX_N) ** exponent


def fit_power_law(n_values: IntArray, seconds: FloatArray) -> PowerLawFit:
    """Fit t = c * n^k to the points with n >= FIT_MIN_N."""
    in_fit_range = n_values >= FIT_MIN_N
    n_fit = n_values[in_fit_range].astype(float)
    t_fit = seconds[in_fit_range]
    # sigma = t makes the residuals relative, so the small-n points count as
    # much as the large-n ones (like a fit on log-log axes).
    (scale_s, exponent), covariance = curve_fit(
        power_law, n_fit, t_fit, p0=(t_fit[-1], INITIAL_EXPONENT_GUESS), sigma=t_fit
    )
    return PowerLawFit(scale_s, exponent, math.sqrt(covariance[1, 1]))


def plot_execution_time(
    n_values: IntArray, fast_s: FloatArray, naive_s: FloatArray
) -> Figure:
    """Linear-axis plot of time vs n: both methods, plus fast doubling alone.

    The second panel exists because on a shared linear axis the fast method
    is flattened against zero by the naive one.
    """
    fig, (ax_both, ax_fast) = plt.subplots(1, 2, figsize=(12, 5))
    ax_both.plot(n_values, naive_s * MS_PER_S, color=NAIVE_COLOR,
                 linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                 label="Naive recurrence")
    ax_both.plot(n_values, fast_s * MS_PER_S, color=FAST_COLOR,
                 linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                 label="Fast doubling")
    ax_both.set_title("Execution time vs n")

    ax_fast.plot(n_values, fast_s * MS_PER_S, color=FAST_COLOR,
                 linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE,
                 label="Fast doubling")
    ax_fast.set_title("Fast doubling only")

    for ax in (ax_both, ax_fast):
        ax.set_xlabel("n")
        ax.set_ylabel("Execution time [ms]")
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_loglog_with_fits(
    n_values: IntArray,
    fast_s: FloatArray,
    naive_s: FloatArray,
    fast_fit: PowerLawFit,
    naive_fit: PowerLawFit,
) -> Figure:
    """Log-log plot of the measurements with the fitted power laws overlaid."""
    positive = n_values > 0  # n = 0 has no place on a log axis
    n_curve = np.geomspace(FIT_MIN_N, BENCHMARK_MAX_N, FIT_PLOT_POINTS)

    fig, ax = plt.subplots(figsize=(8, 6))
    series = (
        ("Naive recurrence", naive_s, naive_fit, NAIVE_COLOR),
        ("Fast doubling", fast_s, fast_fit, FAST_COLOR),
    )
    for name, seconds, fit, color in series:
        ax.plot(n_values[positive], seconds[positive] * MS_PER_S,
                linestyle="none", marker="o", markersize=MARKER_SIZE,
                color=color, label=f"{name} (measured)")
        ax.plot(n_curve, power_law(n_curve, fit.scale_s, fit.exponent) * MS_PER_S,
                linestyle="--", linewidth=LINE_WIDTH, color=color,
                label=f"{name} fit: k = {fit.exponent:.3f}")
    ax.axvline(FIT_MIN_N, color="gray", linestyle=":", linewidth=1,
               label=f"Fit range starts (n = {FIT_MIN_N})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Execution time [ms]")
    ax.set_title("Execution time vs n (log-log) with fitted t = c·n^k")
    ax.grid(True, which="both", alpha=GRID_ALPHA)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, filename: str) -> None:
    """Save a figure as a PNG in PLOTS_DIR, creating the folder if needed."""
    PLOTS_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")


def print_value_summary(n: int, value: int) -> None:
    """Print the digit count and the leading/trailing digits of S_n."""
    num_digits, leading, trailing = summarize_digits(value, EDGE_DIGITS)
    print(f"S_{n} has {num_digits} digits: {leading}...{trailing}")


def print_benchmark_summary(
    n_values: IntArray,
    fast_s: FloatArray,
    naive_s: FloatArray,
    fast_fit: PowerLawFit,
    naive_fit: PowerLawFit,
) -> None:
    """Print the timings at the largest n and the fitted exponents."""
    n_max = n_values[-1]
    print(f"Time at n = {n_max}: fast doubling {fast_s[-1] * MS_PER_S:.3f} ms, "
          f"naive {naive_s[-1] * MS_PER_S:.1f} ms")
    print(f"Fitted k (n >= {FIT_MIN_N}): "
          f"fast doubling {fast_fit.exponent:.3f} ± {fast_fit.exponent_std:.3f} "
          f"(Karatsuba theory {KARATSUBA_EXPONENT:.3f}), "
          f"naive {naive_fit.exponent:.3f} ± {naive_fit.exponent_std:.3f} "
          "(theory 2)")


def main() -> None:
    verify_against_naive(VERIFY_MAX_N)
    print(f"sequence_calculator matches naive_sequence for n < {VERIFY_MAX_N}")
    print_value_summary(TARGET_N, sequence_calculator(TARGET_N))

    n_values = np.linspace(0, BENCHMARK_MAX_N, BENCHMARK_NUM_POINTS).astype(np.int64)
    fast_s = benchmark(sequence_calculator, n_values, FAST_REPEATS, FAST_CALLS_PER_REPEAT)
    naive_s = benchmark(naive_sequence, n_values, NAIVE_REPEATS, NAIVE_CALLS_PER_REPEAT)
    fast_fit = fit_power_law(n_values, fast_s)
    naive_fit = fit_power_law(n_values, naive_s)
    print_benchmark_summary(n_values, fast_s, naive_s, fast_fit, naive_fit)

    save_figure(plot_execution_time(n_values, fast_s, naive_s), "sequence_time.png")
    save_figure(
        plot_loglog_with_fits(n_values, fast_s, naive_s, fast_fit, naive_fit),
        "sequence_loglog.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
