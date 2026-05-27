# CODEX HANDOFF STAGE34

## 1. Current Goal

The project is DCASE Task 2 / ASD. The user's long-term target is to raise mean hmean toward `0.7`.

Current route family:

- frozen clean/adapted EAT embeddings
- backend remains frozen: `min_clamp__mean + ldn_diff_K1_scorek5`
- no selector, no fusion, no LDN `K/r/score_k` tuning
- no anomaly labels for training
- no test-label checkpoint selection

The latest productive direction is representation-level pseudo-attribute/domain-adaptive training. Pure pooling, raw segment scoring, fixed prototypes, and simple guardrail scheduling did not produce enough improvement.

## 2. Current Route Hierarchy

### Stage30: current hmean best

Route:

`Stage30 guardrailed pseudo-attribute EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`

Checkpoint:

`results/stage30a_guardrailed_pseudo_attribute/checkpoints/stage30a_guardrailed_pseudo_attribute_best.pt`

Full7 result:

- mean_hmean: `0.6231264650860286`
- mean_pAUC: `0.5717293233082706`
- mean_TPR99: `0.10857142857142857`
- delta_hmean_vs_clean_best: `+0.007179736243476897`
- delta_pAUC_vs_clean_best: `+0.011428571428571344`
- delta_TPR99_vs_clean_best: `+0.0028571428571428498`
- strong_machine_severe_collapse: `false`
- freeze status: `new_hmean_best_with_pauc_tpr_tradeoff`

Important Stage30 interpretation:

- Stage30 is the hmean best.
- It recovered fan versus Stage29.
- It controlled representation drift: Stage30 final `val_anchor_loss = 0.06721972126610341`, compared with Stage29 `0.83134942405162`.
- It gave back Stage29's TPR@99n tail advantage.

### Stage29: pAUC / TPR@99n best

Route:

`Stage29 pseudo-attribute representation-adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`

Checkpoint:

`results/stage29a_pseudo_attribute_representation/checkpoints/stage29a_pseudo_attribute_best.pt`

Full7 result:

- mean_hmean: `0.6171391941988533`
- mean_pAUC: `0.5724060150375939`
- mean_TPR99: `0.12142857142857143`
- delta_hmean_vs_clean_best: `+0.0011924653563015752`
- delta_pAUC_vs_clean_best: `+0.012105263157894619`
- delta_TPR99_vs_clean_best: `+0.01571428571428571`
- freeze status: `new_balanced_best_with_fan_regression`

Important Stage29 interpretation:

- Stage29 is still the pAUC / TPR@99n best.
- It moved geometry too much and regressed fan.
- Stage29 final `val_anchor_loss = 0.83134942405162`.

### Stage27: previous balanced reference

Route:

`Stage27 TF-patch representation-adapted EAT + frozen min_clamp__mean + ldn_diff_K1_scorek5`

Full7 result:

- mean_hmean: `0.6168603949659727`
- mean_pAUC: `0.5624060150375939`
- mean_TPR99: `0.11857142857142856`
- freeze status: `balanced_positive_freeze`

Stage27 was the first balanced positive freeze, but Stage29/30 superseded it.

### Clean best reference

Route:

`EAT min_clamp__mean + ldn_diff_K1_scorek5`

Reference metrics:

- mean_hmean: `0.6159467288425517`
- mean_pAUC: `0.5603007518796993`
- mean_TPR99: `0.10571428571428572`

## 3. Negative / Not Promoted Branches

### Stage28: pooling/spike audit

Best route:

`minclamp_rdp_g2__ldn_diff_K1_scorek5`

Result:

- mean_hmean: `0.6159997909609221`
- mean_pAUC: `0.5579699248120301`
- mean_TPR99: `0.09571428571428572`
- freeze status: `weak_hmean_probe_not_promoted`

Conclusion:

RDP pooling exposed a tiny hmean signal but lowered pAUC/TPR99 and stayed below Stage27/29/30. Do not tune pooling exponents or spike thresholds as a score search.

### Stage31: lower-guardrail sweep

Variants:

- `stage31_anchor030`
- `stage31_anchor020`
- `stage31_anchor030_pair025`

Best hmean:

- `0.6195363914140675`, below Stage30 by `-0.0035900736719611803`

Best TPR99:

- `0.10142857142857142`, below Stage30 by `-0.007142857142857145`

Freeze status:

`over_guarded_negative_freeze`

Conclusion:

Lowering guardrail weights made the objective over-constrained. Validation anchor loss fell to about `0.010`, pseudo-attribute val accuracy collapsed to `0`, and tail did not recover.

### Stage32: scheduled guardrail

Design:

- warmup phase: light guardrail
- guard phase: Stage30-like guardrail
- checkpoint selection only in guard phase by normal validation loss

Result:

- mean_hmean: `0.6221673973272781`
- mean_pAUC: `0.568345864661654`
- mean_TPR99: `0.10571428571428572`
- delta_hmean_vs_stage30: `-0.0009590677587505425`
- delta_TPR99_vs_stage30: `-0.0028571428571428498`
- freeze status: `scheduled_guardrail_not_promoted`

Conclusion:

Warmup was underpowered. Pseudo-attribute validation accuracy stayed at `0`, so it did not recover tail.

### Stage33: fixed prototype pseudo-attribute

Design:

- no trainable attribute head
- fixed clean-EAT pseudo-attribute prototypes
- train last EAT block toward prototypes under Stage30-like guardrails

Result:

- mean_hmean: `0.618122874722881`
- mean_pAUC: `0.5626315789473685`
- mean_TPR99: `0.10142857142857142`
- delta_hmean_vs_stage30: `-0.005003590363147681`
- final validation prototype accuracy: `0.0`
- freeze status: `fixed_prototype_underpowered_negative_freeze`

Conclusion:

Fixed prototypes were too weak under guardrails. Do not continue fixed-prototype alignment as implemented.

## 4. Important Diagnosis So Far

1. The original clean frozen route was already strong but near a local ceiling.
2. Stage22 geometry objective gave only a weak hmean improvement and hurt pAUC/TPR99.
3. Fan harm is not a single loss-component problem. It is a density-recalibration mismatch under adapted geometry.
4. Local-head-only / segment scoring routes did not transfer to real anomaly evidence.
5. Pseudo-attribute representation is currently the most productive direction:
   - Stage29 improves pAUC and TPR99.
   - Stage30 guardrails improve hmean and recover fan.
6. Recent failures show that:
   - lowering guardrails is not enough,
   - simple warmup scheduling is not enough,
   - fixed prototypes are too weak.

## 5. Key Files / Scripts

Current relevant scripts:

- `scripts/audit_stage29a_pseudo_attribute_representation.py`
- `scripts/audit_stage29b_pseudo_attribute_freeze.py`
- `scripts/audit_stage30a_guardrailed_pseudo_attribute.py`
- `scripts/audit_stage30b_guardrailed_freeze.py`
- `scripts/audit_stage31a_guardrail_sweep_summary.py`
- `scripts/audit_stage31b_guardrail_sweep_freeze.py`
- `scripts/audit_stage32a_scheduled_guardrail_pseudo_attribute.py`
- `scripts/audit_stage32b_scheduled_guardrail_freeze.py`
- `scripts/audit_stage33a_fixed_prototype_representation.py`
- `scripts/audit_stage33b_fixed_prototype_freeze.py`
- `scripts/audit_stage22d_adapted_full7_eval.py`

Important result folders:

- `results/stage29a_pseudo_attribute_representation`
- `results/stage29a_pseudo_attribute_full7_eval`
- `results/stage29b_pseudo_attribute_freeze`
- `results/stage30a_guardrailed_pseudo_attribute`
- `results/stage30a_guardrailed_pseudo_attribute_full7_eval`
- `results/stage30b_guardrailed_freeze`
- `results/stage31b_guardrail_sweep_freeze`
- `results/stage32b_scheduled_guardrail_freeze`
- `results/stage33b_fixed_prototype_freeze`

Pseudo-attribute inventory:

`results/stage29a_pseudo_attribute_representation/pseudo_attribute_inventory.csv`

Clean EAT train embeddings used for prototypes:

`results/stage29a_pseudo_attribute_representation/*_clean_eat_train_embeddings.npz`

## 6. Operating Preferences

The user runs training on the server. Codex should write local functionality and provide one-line commands.

Server working directory:

`/data1/test/LAES_2025/new_projects/clean_tensor_asd`

Default GPU:

`CUDA_VISIBLE_DEVICES=1`

Command style:

- Prefer one-line commands.
- Do not train inside Codex.
- It is OK for Codex to use SSH/SCP for light checks, file sync, and reading result files.

## 7. Forbidden Moves

Do not:

- tune LDN `K/r/score_k`,
- add selector,
- add fusion to choose routes per machine,
- use anomaly labels for training,
- use test labels for checkpoint selection,
- keep stretching the same warmup schedule without evidence of pseudo-attribute learning,
- continue fixed-prototype alignment as implemented,
- tune pooling or prototype hyperparameters as open-ended score search.

## 8. Recommended Stage34

The next useful branch should return to Stage30-near settings but add a normal-only tail/dispersion objective.

Rationale:

- Stage30 gives the best hmean and recovers fan.
- Stage29 gives the best tail but moves geometry too much.
- Stage31/32/33 show that guardrail schedule/prototype changes do not restore tail.
- Therefore, the next objective should directly protect normal score tail / density geometry rather than trying to recover tail indirectly through pseudo-attribute accuracy.

Suggested Stage34 direction:

`Stage30 guardrailed pseudo-attribute + normal-only score-tail/dispersion regularizer`

Possible normal-only regularizer:

- compute student clean clip embeddings and teacher clean clip embeddings within batch,
- preserve teacher local-density ordering or batch kNN distance distribution,
- penalize excessive compression/expansion of high-quantile normal distances,
- keep checkpoint compatible with Stage22D frozen backend.

Stage34 success criterion:

- mean_hmean should stay close to or exceed Stage30 `0.623126`,
- mean_TPR99 should improve over Stage30 `0.108571`,
- fan should not regress toward Stage29,
- no strong-machine severe collapse.

Initial Stage34 implementation advice:

- Start from `scripts/audit_stage30a_guardrailed_pseudo_attribute.py`.
- Add a teacher-vs-student normal-tail geometry loss, not another head.
- Keep default Stage30 weights: `anchor=0.50`, `pairwise=0.50`, `center=0.25`.
- Add a small tail loss weight first, e.g. `0.10`.
- Save `student_state_dict` in the same Stage21E/22D-compatible format.

