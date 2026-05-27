# CTAF V16 To V29 Freeze Summary

## Scope

This note freezes the decision state after:

- `V16` multi-seed audit
- `V27` and `V28` protocol-safe diagnostics
- `V29` multi-resolution structure test

The purpose is to prevent drifting back into single-seed "new best" decisions.

## Frozen Mainline

Current mainline architecture:

```text
CTAF V16 =
full_5view tensor frontend
+ viewwise_unshared_late_fusion Conv2D AE
+ latent summary vectors(mean/std/max)
+ train-normal latent kNN memory score
+ k=5
+ source-only normal training
```

## What Is Stable

`V16` is directionally stable:

- `full_5view` beats matched `logmel_only` on hmean in `5/5` seeds
- `full_5view` beats matched `logmel_only` on both hmean and pAUC in `4/5`
  seeds

So the CTAF direction itself is real, not a pure accident.

## What Is Not Stable

The original frozen V16 single-seed point is not stable enough to remain the
only hard baseline.

Original frozen point:

```text
hmean = 0.564294
pAUC  = 0.553459
KS    = 0.2886
seed  = 13711
```

Direct rerun of the same seed:

```text
hmean = 0.553511
pAUC  = 0.547820
KS    = 0.3286
seed  = 13711
```

Therefore the correct operational baseline is now the V16 seed-panel mean:

```text
hmean = 0.549547
pAUC  = 0.542331
KS    = 0.294880
```

## V27-V29 Status

- `V27`: diagnostic only, not a replacement
- `V28`: diagnostic only, not a replacement
- `V29`: meaningful structure test, beats matched `logmel_only`, but does not
  beat V16 strongly enough to replace it

So:

```text
V16 remains the active mainline, but the seed-panel mean is the effective
decision baseline.
```

## New Evaluation Rules

### A. Matched Baseline Gate

Any new structure must first beat the matched `logmel_only` baseline.

### B. Seed-Panel Mean Gate

A single-seed structural run must at least exceed:

```text
hmean > 0.549547
pAUC  >= 0.542331
KS not clearly worse than 0.294880
```

If it cannot pass this gate, it is not worth treating as a serious successor
candidate.

### C. Candidate Gate

If a single-seed run exceeds the original frozen V16 point:

```text
hmean > 0.564294
pAUC  >= 0.553459
```

it may be labeled only:

```text
candidate replacement
```

It is not allowed to become the new frozen best on a single seed.

### D. Freeze Gate

A new structure can replace V16 only if:

- its `3-5` seed mean exceeds the V16 seed-panel mean
- most of its seeds beat matched `logmel_only`
- `KS` is not clearly degraded

## Operational Rule

From this point onward:

```text
Do not declare a new best method from a single seed.
Do not resume broad automatic structure search without applying the seed-panel rule.
```

## Next-Step Constraint

The next structure innovation, if any, should be launched in a controlled way:

- do not open many versions at once
- use the matched baseline gate first
- if promising, escalate to `3-5` seed validation
- only after multi-seed confirmation can the method challenge V16
