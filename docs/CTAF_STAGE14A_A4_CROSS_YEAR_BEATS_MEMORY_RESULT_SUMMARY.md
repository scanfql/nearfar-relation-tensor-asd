# CTAF Stage14A A4 Cross-Year BEATs Memory Result Summary

## 1. B2s Primary Result

Reference `A0 raw BEATs`:

- `hmean = 0.5785`
- `pAUC = 0.5388`
- `TPR@99n = 0.0500`

Primary `B2s = source-only cross-year frozen BEATs memory`:

- `hmean = 0.5773`
- `pAUC = 0.5421`
- `TPR@99n = 0.0557`

Delta vs `A0`:

- `Δ hmean = -0.0012`
- `Δ pAUC = +0.0033`
- `Δ TPR@99n = +0.0057`

Interpretation:

`B2s` does **not** improve `hmean` over `A0`, so it is **not** a full positive
route.

However:

- `pAUC` improves slightly
- `TPR@99n` improves slightly

Therefore `B2s` should be described as:

`weak positive / mixed`

not as useless, and not as a new frozen replacement route.

Mechanistically, this means:

- source-only cross-year frozen BEATs memory gives a small stability benefit
- but not a strong enough global-ranking gain to justify replacing raw BEATs

In other words:

`more source-only normal support helps, but only modestly.`

## 2. F2s Primary Fusion Result

Reference `A1 = V63 + raw BEATs fixed rank fusion`:

- `hmean ≈ 0.5813`
- `pAUC ≈ 0.5484`
- `TPR@99n ≈ 0.0976`

Primary `F2s = V63 + source-only cross-year frozen BEATs`:

- `hmean ≈ 0.5774`
- `pAUC ≈ 0.5469`
- `TPR@99n ≈ 0.0933`

Delta vs `A1`:

- `Δ hmean ≈ -0.0039`
- `Δ pAUC ≈ -0.0015`
- `Δ TPR@99n ≈ -0.0043`

Interpretation:

`F2s` does **not** beat `A1`.

It is therefore:

- a useful source-only diagnostic
- but **not** a new external frozen route
- and it should **not** replace `A1`

The failure here is important:

- source-only cross-year BEATs memory does not preserve the full `A1` gain
- the external route still benefits more from raw current-year BEATs than from
  source-only memory expansion alone

## 3. B2f Diagnostic Result

Diagnostic `B2f = full-manifest frozen BEATs memory`:

- `hmean = 0.5842`
- `pAUC = 0.5414`
- `TPR@99n = 0.0600`

Relative to `B2s`:

- `hmean` is clearly higher
- `TPR@99n` is also higher
- `pAUC` is very similar, slightly lower

Interpretation:

`B2f` shows a stronger `hmean` signal than `B2s`, suggesting that
target-domain normal memory contributes meaningful support.

But `B2f` is **diagnostic only**, because it includes target train-normal
memory.

So it can be used to support the statement:

`target-domain memory effect is real`

but not to define the primary route.

This is the most important diagnostic conclusion from A4:

`the strongest memory-bank gain is not purely cross-year source-only; it appears to depend partly on target-domain train-normal support.`

## 4. F2f Diagnostic Fusion Result

Diagnostic `F2f = V63 + full-manifest BEATs`:

- `hmean ≈ 0.5816`
- `pAUC ≈ 0.5466`
- `TPR@99n ≈ 0.0938`

Relative to `A1`:

- `hmean` is slightly higher
- `pAUC` is lower
- `TPR@99n` is lower

Important interpretation rule:

Do **not** write that `F2f` beats `A1` on `hmean`, `pAUC`, and `TPR@99n`.

Correct interpretation:

- `F2f` is slightly higher than `A1` on `hmean`
- but lower than `A1` on `pAUC`
- and lower than `A1` on `TPR@99n`

Therefore:

`F2f` provides diagnostic `hmean` signal, but it is not a balanced replacement for `A1`.

So the right interpretation is:

- `F2f` is an informative upper-support diagnostic
- but not a new frozen route
- because it fails to dominate `A1` on the balance of `hmean + pAUC + TPR@99n`

## 5. Machine-Level Findings

Key machine-level observations:

- `fan` remains the main collapse point
- `bearing` still favors `V63 / A1` more than the BEATs-memory routes
- `gearbox / slider / toycar / toytrain / valve` show BEATs-memory `hmean` support
- full-manifest gains on `toycar / toytrain` suggest a target-domain or extra
  normal-support effect

More specifically:

- `fan` degrades strongly under both `F2s` and `F2f`, so it is the clearest
  argument against promoting A4 to main-route status
- `bearing` remains better aligned with the CTAF-dominant side (`V63 / A1`) than
  with BEATs-memory-heavy variants
- `gearbox` benefits from memory support, but not enough to offset the broader
  tradeoff
- `slider` shows some hmean support but not enough low-FPR gain
- `toycar` and `toytrain` improve more clearly under the full-manifest
  diagnostic branch than under source-only memory, reinforcing the
  target-support interpretation
- `valve` remains a strong machine for the BEATs branch in almost every setup

This means the memory expansion story is not uniform:

- some machines benefit from more external normal support
- some machines, especially `fan`, remain resistant
- the strongest gains are not distributed evenly enough to justify route
  replacement

So A4 is informative as a mechanistic branch, but not yet strong enough to
change the frozen external route decision.

## 6. Final Verdict

`A4 primary source-only route:`

- `B2s = weak positive / mixed`
- `F2s = FAIL as A1 replacement`

`A4 full-manifest diagnostic:`

- `B2f = diagnostic hmean-positive`
- `F2f = diagnostic hmean signal, not balanced route replacement`

Current best external route remains:

- `A1 fixed V63 + raw BEATs rank fusion`

In summary:

- `A4 primary = mixed / not full success`
- `A4 diagnostic = target-domain memory effect is real, but does not replace A1`

## 7. Blocked Actions

Do not:

- replace `A1` with `A4`
- use `B2f / F2f` as primary
- claim `F2f` dominates `A1`
- add target memory to the main route without preregistration
- tune memory composition
- tune fusion weights
- do per-machine selection
- fine-tune BEATs
- open Stage 14B automatically

This keeps the A4 result properly bounded:

- informative
- mechanistically useful
- but not a route replacement
