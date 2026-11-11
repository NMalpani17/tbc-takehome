# TBC Team take-home

Two independent Python assignments:

| File | Assignment |
|------|------------|
| `car_simulator.py` | Kinematic bicycle-model car simulator: straight acceleration, then a clockwise 100 m turn |
| `sequence.py` | Fast computation of S_n = 3·S_(n-1) − S_(n-2), with complexity analysis and benchmark |

## Setup and running

Requires Python 3.12+ and the packages in `requirements.txt` (numpy, matplotlib, scipy).

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python car_simulator.py
python sequence.py               # runs the benchmark, ~20 s
```

Each script prints a short summary, saves its figures to `plots/` and opens them in a window.

---

## 1. `car_simulator.py`

**Model.** Kinematic bicycle model, rear axle as the reference point, integrated with explicit Euler.
All derivatives are taken from the current state before any state variable is updated:

```
x     += v·cos(θ)·dt
y     += v·sin(θ)·dt
θ     += v/L·tan(δ)·dt        L = wheelbase, δ = wheel angle
v     += a·dt
```

**Scenario.** Start at rest and accelerate at 0.4 m/s² in a straight line until 10 m/s. The final step's
acceleration is clamped, so the speed cannot overshoot. Then hold that speed with a constant wheel angle
δ = −atan(L / 100 m), which gives a clockwise circle of radius 100 m. The run ends at t = 100 s, with dt = 0.1 s.
The controller, which picks `a` and `δ` from the current speed, is a separate function from the simulation loop.

**Results.**
- 10 m/s is reached at **t = 25.0 s**, at **(x, y) = (124.50, 0.00) m**. The exact answer for constant
  acceleration is 125 m. Euler moves each step at the speed from the start of that step, so it covers
  slightly less distance.
- Steady-state lateral acceleration: **−1.000 m/s²** (= −v²/R; negative means towards the right).
- With constant v and δ, every Euler step has the same length and the same heading change. The turn is
  therefore a regular polygon whose vertices lie on a circle of radius ≈ 100.0004 m, so the path closes
  on itself instead of spiralling outward.

Figures: `plots/trajectory.png`, `plots/accelerations.png`.

---

## 2. `sequence.py`

### Key identity: S_n = F(2n)

The sequence 0, 1, 3, 8, 21, 55, … is every other Fibonacci number. Applying F(m) = F(m−1) + F(m−2) twice:

```
F(2n) = F(2n−1) + F(2n−2)
      = 2F(2n−2) + F(2n−3)
      = 2F(2n−2) + (F(2n−2) − F(2n−4))
      = 3F(2n−2) − F(2n−4)
```

So F(2n) satisfies the same recurrence as S_n, with the same initial values (F(0) = 0, F(2) = 1).
By induction, S_n = F(2n).

### Algorithm

Iterative fast-doubling Fibonacci over the bits of n:

```
F(2k)   = F(k)·(2F(k+1) − F(k))
F(2k+1) = F(k)² + F(k+1)²
```

After bit_length(n) iterations the loop holds (F(n), F(n+1)). One more doubling step gives F(2n) = S_n.
The results are checked against the plain recurrence (`naive_sequence`) for all n < 200.

S_100000 has **41,798 digits**: `15085683557988938992…52246259408175853125`.

### Time complexity: O(n^log₂3) ≈ O(n^1.585)

The loop runs only O(log n) times, but **O(log n) is not the running time**, because the operands are huge.

1. **Size of the numbers.** S_n = F(2n) ≈ φ^(2n)/√5, so S_n has about 2·log₂(φ)·n ≈ **1.39n bits**
   (≈ 0.42n decimal digits). The operands of the final multiplication, F(n) and F(n+1), have ≈ 0.69n bits.
2. **Cost of one multiplication.** CPython multiplies large integers with **Karatsuba**:
   M(b) = O(b^log₂3) = O(b^1.585) for b-bit operands. Additions, subtractions and shifts are only O(b).
3. **Summing over iterations.** The operands double in size every iteration. Each earlier iteration
   therefore works on numbers half as large and costs 2^1.585 = 3 times less. The total is a geometric
   series dominated by the final multiplications:

   ```
   M(b)·(1 + 1/3 + 1/9 + …) = 1.5·M(b),   b ≈ 0.69n   →   O(n^1.585)
   ```

**Comparison, naive loop:** n iterations, each an O(b) addition on numbers of up to ~1.39n bits,
gives O(n²).

The benchmark times only the computation of S_n. Converting S_n to a decimal string has its own cost and
is not part of computing S_n.

### Benchmark and curve fit

41 evenly spaced values of n from 0 to 100,000 (step 2,500). For each n the script runs several `timeit` repeats
with garbage collection disabled and keeps the **minimum**, since OS noise only ever adds time.
`t = c·n^k` is then fitted with `scipy.optimize.curve_fit` on n ≥ 5,000. Below that, the timings are
mostly interpreter overhead. The fit uses relative residuals, so small-n and large-n points carry equal weight.

Results on the development machine (Windows 11, CPython 3.12.5; absolute times vary by machine):

| Method | Time at n = 100,000 | Fitted k | Theory |
|--------|--------------------:|---------:|-------:|
| Fast doubling | **1.73 ms** | **1.641 ± 0.013** | 1.585 (Karatsuba) |
| Naive recurrence | **339 ms** | **1.931 ± 0.008** | 2 |

At n = 100,000 fast doubling is about 200× faster, and the gap grows with n.

Figures: `plots/sequence_time.png` (linear axes, plus a fast-doubling-only panel because the naive
curve flattens it against zero) and `plots/sequence_loglog.png` (log-log with the fitted curves).

### Why the fitted exponents differ from theory

- **Fast doubling (1.64 vs 1.585).** O(b^1.585) is an asymptotic bound. CPython uses Karatsuba only for
  operands above roughly 70 internal digits (~2,100 bits). Below that cutoff it uses O(b²) schoolbook
  multiplication. In the tested range the largest operands are ~70,000 bits, so the Karatsuba recursion
  is only a few levels deep before it reaches quadratic base cases. That makes the effective exponent a
  little higher than log₂3, and it approaches 1.585 only as n grows. Memory allocation and cache effects
  on the larger operands add a little more. The jitter near n ≈ 90,000–100,000 is within run-to-run noise
  at these millisecond timings. The cost may also not be perfectly smooth in n, because Karatsuba splits
  numbers at different points depending on their exact size.
- **Naive (1.93 vs 2).** Each iteration also pays a fixed interpreter cost (loop bookkeeping, tuple
  packing, allocating the new int object). That is an O(n) term added to the O(n²) arithmetic, and it
  pulls the fitted exponent slightly below 2, mostly through the smaller n values in the fit range.

Both measured exponents are close to the predicted ones, and they are clearly separated from each other.
The data rules out O(log n) for fast doubling and confirms that the naive loop is about quadratic.
