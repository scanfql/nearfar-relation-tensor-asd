# CTAF V39 Frontend Pre-screen Report

## 1. Pre-registered Gate Definition

All gates were written **before** running this diagnostic. No threshold was tuned per machine.

| Gate | Condition | Threshold |
|------|-----------|-----------|
| A | robust_d_missed >= pass_robust_d | >= 0.3 |
| B | >= min_pass_machines machines pass Gate A | >= 2 |
| C | max_redundancy_r <= max_redundancy_r | <= 0.85 |
| D | robust_d_detected > detected_sanity_min | > 0.0 |

robust_d = (median(group) - median(normal_tail)) / (0.5*(IQR(group)+IQR(normal_tail)) + 1e-8)

## 2. Reference Score Source

- Result directory: `results/v37_teacher_student_pretrain_seedpanel/seed_13711`
- Experiment: `full_5view`
- Score mode: `latent_summary_knn`
- Normal-tail band: q0.95 -- q0.99

## 3. Sample Group Statistics

| Machine | n_normal_tail | n_missed | n_detected | n_total_anomalies | q_low | q_high |
|---------|---------------|----------|------------|-------------------|-------|--------|
| fan | 4 | 88 | 12 | 100 | 0.229054 | 0.238505 |
| gearbox | 4 | 94 | 6 | 100 | 0.117578 | 0.129624 |
| slider | 4 | 100 | 0 | 100 | 0.122654 | 0.159626 |
| toycar | 4 | 96 | 4 | 100 | 0.077860 | 0.082011 |
| toytrain | 4 | 97 | 3 | 100 | 0.095072 | 0.122015 |

## 4. Per-Frontend Per-Machine Separation

| Machine | Frontend | robust_d_missed | robust_d_detected | cohens_d | Gate A | Gate D | Flags |
|---------|----------|-----------------|-------------------|----------|--------|--------|-------|
| fan | cepstral_3view | 0.0000 | 0.0000 | 0.0000 | FAIL | FAIL | nan_in_tail_distances; nan_in_missed_distances |
| fan | modulation_3view | 2.0524 | 7.1668 | 2.1273 | PASS | PASS | -- |
| fan | transient_3view | 1.5832 | 2.1515 | 1.9408 | PASS | PASS | -- |
| gearbox | cepstral_3view | 0.0000 | 0.0000 | 0.0000 | FAIL | FAIL | nan_in_tail_distances; nan_in_missed_distances |
| gearbox | modulation_3view | 2.5461 | 1.9562 | 1.7517 | PASS | PASS | -- |
| gearbox | transient_3view | 0.5006 | 0.7382 | 0.5442 | PASS | PASS | -- |
| slider | cepstral_3view | 0.0000 | 0.0000 | 0.0000 | FAIL | FAIL | no_detected_anomalies; nan_in_tail_distances; nan_in_missed_distances |
| slider | modulation_3view | 1.8682 | 0.0000 | 2.1887 | PASS | FAIL | no_detected_anomalies |
| slider | transient_3view | 0.5092 | 0.0000 | 0.6835 | PASS | FAIL | no_detected_anomalies |
| toycar | cepstral_3view | 0.0000 | 0.0000 | 0.0000 | FAIL | FAIL | nan_in_tail_distances; nan_in_missed_distances |
| toycar | modulation_3view | 8.6973 | 11.1008 | 10.3587 | PASS | PASS | -- |
| toycar | transient_3view | 0.6029 | 2.6810 | 1.3694 | PASS | PASS | -- |
| toytrain | cepstral_3view | 0.0000 | 0.0000 | 0.0000 | FAIL | FAIL | nan_in_tail_distances; nan_in_missed_distances |
| toytrain | modulation_3view | 3.0240 | 9.8991 | 3.5049 | PASS | PASS | -- |
| toytrain | transient_3view | 0.4937 | 0.7582 | 0.7464 | PASS | PASS | -- |

## 5. Redundancy (Pearson r vs 5 baseline views)

| Frontend | View | max(|r|) |
|----------|------|----------|
| modulation_3view | modulation_low | 0.9688 |
| modulation_3view | modulation_high | 0.6331 |
| cepstral_3view | cepstral_low_quefrency | 0.6089 |
| cepstral_3view | cepstral_periodicity | 0.0000 |
| transient_3view | spectral_flux | 0.0538 |
| transient_3view | transient_energy | 0.0459 |

## 6. Gate Results Table

| Frontend | Machines Pass A | Gate B (>=2) | max_r | Gate C (<=0.85) | Machines Pass All | Stage 0 |
|----------|-----------------|--------------|-------|----------------|-------------------|---------|
| cepstral_3view | 0/5 | FAIL | 0.6089 | PASS | 0/5 | FAIL |
| modulation_3view | 5/5 | PASS | 0.9688 | FAIL | 0/5 | FAIL |
| transient_3view | 5/5 | PASS | 0.0538 | PASS | 4/5 | PASS |

## 7. Per-Machine Visual Examples

See `results/v39_frontend_prescreen/figures/` for example panels.

## 8. Final Verdict

**Stage 0 clearance granted** for the following frontends:
- `transient_3view`

These frontends satisfy all four pre-registered gates and are candidates for V40 train-from-scratch AE evaluation.

---

This diagnostic is offline-only. No model was trained, no score was tuned. All gates are pre-registered and applied uniformly across machines.
