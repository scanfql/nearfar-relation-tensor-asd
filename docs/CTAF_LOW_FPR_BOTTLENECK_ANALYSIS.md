# CTAF Low-FPR Bottleneck Analysis

## Scope

This note analyzes why `V30` and `V31` show real `hmean` signal but still fail
to clear the active low-FPR gate.

Compared systems:

- `V16`: `full_5view + viewwise_unshared_late_fusion`
- `V30`: `full_5view + viewwise_unshared_temporal_pyramid`
- `V31`: `full_5view + viewwise_unshared_residual_temporal_pyramid`

Reference metrics:

```text
V16 seed-panel mean:
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880

V30 seed 13711:
hmean = 0.562034
pAUC  = 0.548045
KS    = 0.2943

V31 seed 13711:
hmean = 0.555806
pAUC  = 0.541654
KS    = 0.2714
```

The key question is:

```text
Why do V30/V31 show useful hmean signal but still fail to deliver a robust
low-FPR advantage over V16?
```

## 1. Per-Machine pAUC Comparison

`full_5view / latent_summary_knn`:

| machine | V16 pAUC | V30 pAUC | V31 pAUC | V30 delta vs V16 | V31 delta vs V16 |
|---|---:|---:|---:|---:|---:|
| bearing | 0.600526 | 0.618947 | 0.625263 | +0.018421 | +0.024737 |
| fan | 0.576842 | 0.590000 | 0.553158 | +0.013158 | -0.023684 |
| gearbox | 0.523158 | 0.511579 | 0.501579 | -0.011579 | -0.021579 |
| slider | 0.545263 | 0.570526 | 0.520526 | +0.025263 | -0.024737 |
| toycar | 0.559474 | 0.504211 | 0.527895 | -0.055263 | -0.031579 |
| toytrain | 0.516842 | 0.497368 | 0.504737 | -0.019474 | -0.012105 |
| valve | 0.552105 | 0.543684 | 0.558421 | -0.008421 | +0.006316 |

Immediate pattern:

- `bearing` is a consistent positive machine for both `V30` and `V31`
- `gearbox` is a chronic low-FPR failure in all three versions
- `toycar` is the biggest `V30` pAUC drag
- `toytrain` is a consistent pAUC drag in both `V30` and `V31`
- `V31` repairs `valve` and partially repairs `toycar`, but loses too much on
  `fan`, `slider`, and `gearbox`

## 2. Which Machines Keep V30/V31 Below The Line?

### V30

The main pAUC deficits versus `V16` are:

- `toycar`: `-0.055263`
- `toytrain`: `-0.019474`
- `gearbox`: `-0.011579`
- `valve`: `-0.008421`

`V30` still clears the single-seed pAUC gate because `bearing`, `fan`, and
`slider` improve enough to offset those losses.

### V31

The main pAUC deficits versus `V16` are:

- `toycar`: `-0.031579`
- `slider`: `-0.024737`
- `fan`: `-0.023684`
- `gearbox`: `-0.021579`
- `toytrain`: `-0.012105`

This is why `V31` misses the active pAUC line by only `0.000677`: it does not
collapse globally, but too many medium-to-bad pAUC machines stack together.

## 3. What Does \"V31 KS Improves But pAUC Does Not\" Mean?

`V31` reduces overall `KS` from `V16 seed-panel mean 0.294880` to `0.2714`,
but `pAUC` still drops below the gate.

This means:

```text
source-normal and target-normal score distributions became better aligned,
but anomaly scores did not move far enough into the extreme right tail.
```

Concrete examples:

- `gearbox`
  - `KS`: `0.26 -> 0.08`
  - `pAUC`: `0.523158 -> 0.501579`
  - interpretation:
    normals align better, but anomalies still sit too close to the normal tail

- `slider`
  - `KS`: `0.18 -> 0.12`
  - `pAUC`: `0.545263 -> 0.520526`
  - interpretation:
    domain geometry improves, but low-FPR separation gets worse

- `toycar`
  - `KS`: `0.16 -> 0.16`
  - `pAUC`: `0.559474 -> 0.527895`
  - interpretation:
    no geometry disaster, but anomaly tail is still not pulled away enough

So `V31` is not failing because normals drift apart. It is failing because
anomalies do not occupy the far-right score tail strongly enough.

## 4. Are Normal Scores Better Aligned While The Anomaly Tail Still Fails?

Yes. The tail statistics make this explicit.

Below, `n99` is the 99th percentile of normal test scores, `an_med` is the
median anomaly score, and `TPR@99n` is the fraction of anomalies that exceed
that 99th-normal threshold.

| machine | model | n99 | anomaly median | TPR@99n |
|---|---|---:|---:|---:|
| gearbox | V16 | 0.1245 | 0.0959 | 0.06 |
| gearbox | V30 | 0.1296 | 0.0983 | 0.03 |
| gearbox | V31 | 0.1280 | 0.1010 | 0.04 |
| slider | V16 | 0.1319 | 0.1084 | 0.09 |
| slider | V30 | 0.1477 | 0.1175 | 0.07 |
| slider | V31 | 0.1663 | 0.1152 | 0.00 |
| toycar | V16 | 0.0757 | 0.0646 | 0.10 |
| toycar | V30 | 0.0781 | 0.0668 | 0.04 |
| toycar | V31 | 0.0914 | 0.0760 | 0.03 |
| toytrain | V16 | 0.1122 | 0.0694 | 0.05 |
| toytrain | V30 | 0.1125 | 0.0701 | 0.04 |
| toytrain | V31 | 0.1345 | 0.0772 | 0.02 |

What this shows:

- the anomaly median is often still well below the normal 99th-percentile
  threshold
- `V30` and `V31` sometimes raise anomaly scores a little, but they also raise
  the normal tail threshold
- once the normal tail rises faster than the anomaly tail, `pAUC` stalls or
  drops even if overall ranking quality still looks usable

This is the clearest in `slider` under `V31`:

```text
V31 slider:
n99    = 0.1663
an_med = 0.1152
TPR@99n = 0.00
```

So `V31` aligns normals better, but it does not create any meaningful anomaly
mass beyond the very-low-FPR decision boundary.

## 5. Is There A High-hmean / Low-pAUC Tradeoff?

Yes, but it is a partial tradeoff, not a universal one.

Examples:

- `V30 toytrain`
  - `hmean`: `0.529068 -> 0.540963`
  - `pAUC`: `0.516842 -> 0.497368`
  - interpretation:
    moderate-rank separation improves, but extreme-right-tail separation gets
    worse

- `V31 slider`
  - `hmean`: `0.529575 -> 0.544658`
  - `pAUC`: `0.545263 -> 0.520526`
  - interpretation:
    global ranking is better enough to raise hmean, but low-FPR discrimination
    collapses

- `V30 bearing`
  - `hmean`: `0.619938 -> 0.643289`
  - `pAUC`: `0.600526 -> 0.618947`
  - interpretation:
    this machine improves on both axes, so the tradeoff is not structural law

So the actual situation is:

```text
V30/V31 do not create a pure hmean-vs-pAUC tradeoff everywhere.
They improve both metrics on some machines, but on several bottleneck machines
they improve middle-rank behavior without improving the anomaly extreme tail.
```

That is enough to raise `hmean`, but not enough to win at low FPR.

## 6. Implication For Next Structure Work

The bottleneck is now specific:

```text
future structure work must target anomaly tail separation, not overall hmean.
```

More concretely:

- do not optimize for \"better average ranking\" alone
- do not treat better `KS` alone as sufficient
- the key failure mode is:
  - normal score distributions become cleaner or more aligned
  - anomaly scores move upward only mildly
  - the rightmost normal tail moves with them
  - `pAUC` therefore fails to improve

Operational conclusion:

- `V30` proves internal temporal multi-scale modeling is a real signal
- `V31` proves residual correction can improve geometry
- neither one solves the low-FPR bottleneck
- the next valid structure idea must explicitly aim to increase anomaly
  occupancy above the normal 95th-99th percentile tail, especially for:
  - `gearbox`
  - `toycar`
  - `toytrain`
  - `slider`

Final summary:

```text
The current CTAF bottleneck is not "more representation power" in the broad
sense. It is failure to pull anomaly scores far enough into the extreme right
tail while keeping the normal tail under control.
```
