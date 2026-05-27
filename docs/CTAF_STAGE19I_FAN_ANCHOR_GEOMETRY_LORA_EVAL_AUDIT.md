# CTAF Stage19I Fan Anchor Geometry LoRA Evaluation Audit

## Summary

Stage19I evaluates the Stage19H fan anchor + distance-geometry LoRA checkpoint with the fixed Stage18-style backend.

## Guardrails

- fan only
- no full7 conclusion
- no selector
- no fusion
- no oracle
- no checkpoint selection by test labels
- no LDN tuning

## Gate Check

- minimum pass: adapted LDN hmean `> 0.5527191284766243`
- strong pass: adapted LDN hmean `>= 0.5820070015781968`
- representation guardrail: adapted baseline should stay within `0.02` of Stage18 baseline

## Verdict

Stage19I should report the fixed comparison against Stage18 and Stage19F references and leave any further decision to the user.
