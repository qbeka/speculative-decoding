# Decisions and failed experiments

Each entry says what was tried, what the evidence showed and what changed. Raw
numbers are in the named files under `runs/`.

## 1. The off-the-shelf draft costs far more than the papers assume

Leviathan et al. report c, the draft cost divided by the target cost, below 0.05.
On this machine at float16 on Metal, the 1.7B target takes about 19 ms per pass and
the 135M draft takes 8 to 12 ms, so c is about 0.5. See
`runs/profile/mps_float16.json`.

The draft is small in parameters but not in depth: it has 30 layers and the target
has 24. Moving it to the CPU does not help (about 8 ms with 2 to 12 threads). About
2.3 ms of that is framework overhead that does not change with layer width, and
most of the rest is small matrix multiplies that do not gain from more threads.

Result: the distilled draft is a shallow one, not only a retrained one.

## 2. float16 does not preserve the target law, so the benchmark uses float32

float16 is about 50 percent faster than float32 on Metal for the target alone. But
the target's distribution at a position changes with how many tokens share a pass,
because rounding changes. Measured over 360 positions on real weights
(`runs/numeric_gap/mps.json`):

| dtype | Median total variation | Largest total variation |
|---|---|---|
| float16 | 0.0027 | 0.0203 |
| float32 | 0.0000047 | 0.000041 |

A sampler that is exact for the distributions it receives cannot fix a target whose
distribution depends on the block size. float16 was dropped as the default and the
benchmark uses float32.
