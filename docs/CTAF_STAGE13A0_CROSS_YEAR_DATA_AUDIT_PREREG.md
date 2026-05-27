# CTAF Stage 13A-0 / V60 — Cross-Year Normal-Only DCASE Data Schema Audit Preregistration

## 1. Motivation

CTAF V39–V59 has already shown that the current `full_5view + latent_summary_knn`
line is the empirical local optimum inside the single-year CTAF pipeline.

Stage 13 does **not** continue:

- frontend replacement
- pooling replacement
- scoring-rule replacement
- hard-normal consistency search
- external frozen BEATs scoring
- proxy classification
- pseudo-anomaly generation

Instead, Stage 13 asks a new question:

```text
Can multi-year normal-only DCASE Task 2 data be safely and uniformly used as
an expanded CTAF pretraining corpus?
```

V60 is **not** a performance experiment.
It does **not** answer whether multi-year data will improve hmean/pAUC.
It answers only:

```text
Can the multi-year normal-only data protocol be defined safely, consistently,
and without leakage?
```

## 2. Why V60 Is Not Another V39–V59 Version Search

V39–V59 searched within the existing CTAF modeling loop:

- same year
- same data support
- modified frontend / pooling / loss / scoring / external representation use

V60 changes none of those modeling components.
It is purely a **data protocol audit**.

So V60 is not:

- a new model version
- a new training objective
- a new scoring method

It is the entrance gate for a new research branch:

```text
Stage 13A: cross-year normal-only data support
```

## 3. Protocol A vs Protocol B

### Protocol A: Current-Year-Only CTAF Baseline

- current single-year CTAF protocol
- one target year only
- same train/test split as current CTAF mainline
- no external normal support

### Protocol B: Multi-Year Normal-Only CTAF Pretraining

- combine multiple DCASE Task 2 years
- use **normal-only** clips for representation pretraining / memory support
- no anomaly clips
- no anomaly labels
- no test labels
- no weak supervision
- no semi-supervised anomaly training

V60 only decides whether Protocol B is safe enough to be defined.

## 4. Years Under Audit

- `2023`
- `2024`
- `2025`
- `2026`

## 5. Fixed Data Principles

- use only normal data
- do not use anomaly clips
- do not use anomaly labels
- do not use test labels
- do not use weak supervision
- do not use semi-supervised anomaly training

## 6. Audit Questions

V60 must determine:

1. whether each yearly root exists
2. which machine types exist in each year
3. how many normal clips exist per machine
4. whether anomaly clips exist and whether they can be strictly filtered
5. whether train / dev / additional training / evaluation-for-training structures are compatible
6. whether source / target / section / domain metadata exist and are compatible
7. whether sample rate and clip length are reasonably compatible
8. whether naming rules are stable enough for a unified parser
9. whether `full_5view` tensor construction is feasible across all usable data
10. whether there is overlap with the current 2024 target machine set
11. which data are usable for normal-only pretraining
12. which data must be excluded
13. whether leakage risk exists
14. whether machine taxonomy mismatches exist
15. whether 2026’s `Emu` / new setting must be treated separately

## 7. Output Requirements

Outputs:

- `results/v60_cross_year_dcase_data_audit/`
- `docs/CTAF_V60_CROSS_YEAR_DCASE_DATA_AUDIT.md`

Optional implementation script:

- `scripts/audit_v60_cross_year_dcase_data.py`

The script, if implemented, must do only:

- directory scanning
- file counting
- metadata parsing
- sample rate / clip length checks
- naming-rule checks
- tensor feasibility checks
- leakage-risk reporting

No training is allowed.

## 8. Required Report Sections

The final report must include:

1. Stage 13A-0 motivation
2. why this is not V39–V59 version search
3. multi-year normal-only protocol definition
4. per-year dataset schema table
5. per-year normal/anomaly filtering table
6. machine type mapping table
7. source/target/section metadata compatibility
8. full_5view tensor feasibility
9. data leakage risk assessment
10. usable normal-only data count
11. excluded data count and reason
12. final verdict:
   - `V60 pass`
   - `V60 limited`
   - `V60 fail`
13. explicit confirmation that no training occurred

## 9. Future Stages (Plan Only)

### Stage 13A-1 / V61

- same encoder as V37
- only change normal data support
- compare current-year-only vs cross-year normal-only pretraining
- no stronger encoder
- no new loss
- no new scoring

### Stage 13B / V62

- only after Stage 13A-1
- use cross-year protocol
- compare original CNN encoder vs stronger Transformer / Conformer encoder

### Stage 13C / V63

- only if A/B remain insufficient
- then consider a new training objective

These are plan-only placeholders.
They are not authorized in V60.

## 10. Forbidden Actions

- do not train any model
- do not write V61/V62/V63 training code
- do not download data automatically
- do not use anomaly labels
- do not mix anomaly clips into normal-only pretraining
- do not use test labels
- do not introduce BEATs
- do not do UACS
- do not do score fusion
- do not do per-machine selection
- do not auto-open V61

## 11. Final Sentence

V60 does not answer whether cross-year data will improve performance. V60 answers only whether multi-year normal-only DCASE Task 2 data can safely enter a unified CTAF protocol without leakage or uncontrolled schema mismatch. No training is authorized in this step.
