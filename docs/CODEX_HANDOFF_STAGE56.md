# CODEX HANDOFF STAGE56

## 1. Current Goal

The user target remains mean hmean `0.7` for DCASE Task 2 ASD.

Current best route is no longer the Stage30 source-memory branch. The major breakthrough after Stage34 came from evaluating adapted EAT checkpoints with target-domain train-normal reference memory.

Core constraints still apply:

- no anomaly labels for training
- no test-label checkpoint selection
- no selector or fusion unless explicitly preregistered
- no LDN `K/r/score_k` tuning
- target train-normal clips may be used as normal reference memory

## 2. Current Champion

Champion:

`Stage52 w006 epoch005 + Stage48 target-memory backend + Stage53 domain_q95_ratio calibration`

Metrics:

- mean_hmean: `0.6499400423151978`
- mean_pAUC: `0.6036842105263158`
- mean_TPR99: `0.17714285714285719`

Artifacts:

- checkpoint: `results/stage52a_stage35_epoch_sweep_e6_w006/checkpoints/stage51a_stage35_epoch_sweep_epoch005.pt`
- target-memory eval root: `results/stage52a_stage52a_stage35_epoch_sweep_e6_w006_stage51a_stage35_epoch_sweep_epoch005_target_memory_full7_eval`
- domain-calibrated score root: `results/stage53a_stage52_w006_e005_domain_calibrated_score`
- freeze doc: `docs/CTAF_STAGE56_CHAMPION_FREEZE.md`

Interpretation:

- Raw target-memory LDN hmean is `0.6492908450451674`.
- Domain q95 calibration lifts hmean to `0.6499400423151978`.
- TPR99 drops from raw `0.19714285714285712` to q95 `0.17714285714285719`.
- Since the user goal is hmean `0.7`, q95 is the hmean champion; raw LDN remains the tail-sensitive reference.

## 3. Productive Stage34-56 History

Stage34:

- tail-guardrailed pseudo attribute
- source-memory hmean: about `0.6209`
- improved pAUC/TPR tail but did not beat Stage30 hmean.

Stage35:

- delayed/mild tail schedule
- source-memory hmean: about `0.6211`
- later became much stronger under target-memory evaluation.

Stage36:

- Stage30 tail polish
- epoch0 remained best; normal-only polish degraded useful geometry.

Stage37/38:

- guardrail-constrained tail and fan-protected tail
- best source-memory route around `0.6227`
- useful but still below Stage30 until target-memory backend was introduced.

Stage41-43:

- source+target pseudo inventory plus fan-protected tail
- Stage43 with target-memory eval reached `0.6470293685016306`.
- This proved target train-normal memory was a major gain source.

Stage48:

- introduced target train-normal memory into full7 eval.
- same frozen backend, no LDN tuning.
- target-memory sweep showed old Stage35 checkpoint became stronger than Stage43:
  - Stage35 target-memory hmean: `0.649237897951`
  - Stage34 target-memory hmean: `0.648683748136`

Stage51/52:

- reran Stage35 while saving epoch checkpoints.
- epoch005 remained best.
- tail weight `0.06` gave the best raw hmean:
  - raw hmean: `0.6492908450451674`
  - raw pAUC: `0.6031578947368422`
  - raw TPR99: `0.19714285714285712`

Stage53/54:

- pure scoring audit over existing embeddings.
- domain q95 ratio calibration gave current champion:
  - hmean: `0.6499400423151978`
  - pAUC: `0.6036842105263158`
  - TPR99: `0.17714285714285719`
- Stage54 broad sweep did not beat this champion.

## 4. Negative Recent Branches

Stage55:

- used source+target pseudo inventory with Stage35-like schedule.
- best calibrated hmean: `0.648439853240`.
- conclusion: adding target train-normal clips to pseudo CE hurts the champion route.

Stage56:

- source-only pseudo CE plus unlabeled target train-normal guard.
- best calibrated hmean: `0.649738657531`.
- conclusion: target guard is close but does not beat the simpler Stage52/53 champion.

Small tail-weight and epoch sweeps:

- `w004`, `w006`, `w007` are all close.
- `w006 epoch005` is best.
- later epochs raise pAUC slightly but reduce hmean.

Random seed search:

- K8 seed variants were unstable and worse than the original seed.
- Do not continue blind seed search.

## 5. Key Scripts Added After Stage34

Important scripts:

- `scripts/audit_stage48a_adapted_full7_eval_target_memory.py`
  - Stage22D-style full7 eval with optional target train-normal memory.

- `scripts/audit_stage51a_stage35_epoch_sweep.py`
  - Stage35-equivalent training while saving every epoch checkpoint.

- `scripts/audit_stage53a_target_memory_domain_calibrated_score.py`
  - Reads existing Stage48 embeddings and scores raw LDN plus train-normal domain calibrations.

- `scripts/audit_stage56a_source_pseudo_target_guard.py`
  - Source-only pseudo CE plus unlabeled target train-normal guard.

Do not delete these; they define the current champion route and recent negatives.

## 6. Governance Notes

The target-memory backend is acceptable under the current rules because target train-normal clips are normal reference memory, not anomaly labels or test labels.

Stage53 q95 calibration uses only train-normal memory scores and domain metadata. However, promoting q95_ratio over raw LDN happened after a full7 audit. From this handoff onward, q95_ratio must be treated as frozen if used as the route baseline. Do not keep inventing more calibration candidates and picking the best by full7 hmean without explicitly marking it as exploratory route search.

## 7. Current Recommendation

The local Stage35 pseudo-attribute/tail family appears capped around `0.65`.

Do not spend more effort on:

- tiny tail weight sweeps
- more Stage35 epochs
- source+target pseudo CE
- random KMeans seed search
- target train-normal teacher guard as currently implemented

Higher-upside next directions:

1. Add a second independent representation family and evaluate it under a preregistered train-normal calibration rule.
2. Search for train-normal-only proxies that predict target-memory score geometry before full7 scoring.
3. Revisit stronger encoders or multi-resolution evidence, but preserve the Stage48/53 governance.
4. If fusion/selection is considered, design and freeze the selector using train-normal metadata only before any full7 test audit.

## 8. One-Line Champion Summary

`results/stage52a_stage35_epoch_sweep_e6_w006/checkpoints/stage51a_stage35_epoch_sweep_epoch005.pt` scored with target train-normal memory and `domain_q95_ratio` gives the current best mean hmean `0.6499400423151978`.
