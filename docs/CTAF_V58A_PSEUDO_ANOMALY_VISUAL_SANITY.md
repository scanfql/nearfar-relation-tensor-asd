# CTAF V58A Pseudo-Anomaly Visual Sanity Check

## 1. Stage 12 Prereg Summary

Stage 12 proposes within-machine pseudo-anomaly discrimination training.
V58A checks whether same-machine local TF patch replacement generates
visually plausible pseudo-anomalies without obvious patch-boundary artifacts.

## 2. Protocol

- patch_time_frac = 0.05
- patch_freq_frac = 0.05
- n_patches = 1
- apply_to_views = all 5
- donor = same-machine source train-normal (different sample)
- n_pairs_per_machine = 5
- seed = 13711

## 3. Per-Machine Visual Examples


### bearing

- pair 0: patch_time=[154,169) patch_freq=[69,75) → `bearing_pair00.png`
- pair 1: patch_time=[44,59) patch_freq=[22,28) → `bearing_pair01.png`
- pair 2: patch_time=[246,261) patch_freq=[1,7) → `bearing_pair02.png`
- pair 3: patch_time=[134,149) patch_freq=[60,66) → `bearing_pair03.png`
- pair 4: patch_time=[261,276) patch_freq=[25,31) → `bearing_pair04.png`

### fan

- pair 0: patch_time=[114,129) patch_freq=[71,77) → `fan_pair00.png`
- pair 1: patch_time=[12,27) patch_freq=[116,122) → `fan_pair01.png`
- pair 2: patch_time=[19,34) patch_freq=[17,23) → `fan_pair02.png`
- pair 3: patch_time=[278,293) patch_freq=[119,125) → `fan_pair03.png`
- pair 4: patch_time=[59,74) patch_freq=[119,125) → `fan_pair04.png`

### gearbox

- pair 0: patch_time=[184,199) patch_freq=[47,53) → `gearbox_pair00.png`
- pair 1: patch_time=[72,87) patch_freq=[62,68) → `gearbox_pair01.png`
- pair 2: patch_time=[100,115) patch_freq=[101,107) → `gearbox_pair02.png`
- pair 3: patch_time=[195,210) patch_freq=[104,110) → `gearbox_pair03.png`
- pair 4: patch_time=[65,80) patch_freq=[32,38) → `gearbox_pair04.png`

### slider

- pair 0: patch_time=[157,172) patch_freq=[63,69) → `slider_pair00.png`
- pair 1: patch_time=[1,16) patch_freq=[17,23) → `slider_pair01.png`
- pair 2: patch_time=[26,41) patch_freq=[115,121) → `slider_pair02.png`
- pair 3: patch_time=[120,135) patch_freq=[13,19) → `slider_pair03.png`
- pair 4: patch_time=[85,100) patch_freq=[16,22) → `slider_pair04.png`

### toycar

- pair 0: patch_time=[334,352) patch_freq=[8,14) → `toycar_pair00.png`
- pair 1: patch_time=[20,38) patch_freq=[8,14) → `toycar_pair01.png`
- pair 2: patch_time=[117,135) patch_freq=[22,28) → `toycar_pair02.png`
- pair 3: patch_time=[87,105) patch_freq=[40,46) → `toycar_pair03.png`
- pair 4: patch_time=[283,301) patch_freq=[16,22) → `toycar_pair04.png`

### toytrain

- pair 0: patch_time=[332,350) patch_freq=[79,85) → `toytrain_pair00.png`
- pair 1: patch_time=[174,192) patch_freq=[85,91) → `toytrain_pair01.png`
- pair 2: patch_time=[120,138) patch_freq=[84,90) → `toytrain_pair02.png`
- pair 3: patch_time=[3,21) patch_freq=[98,104) → `toytrain_pair03.png`
- pair 4: patch_time=[11,29) patch_freq=[50,56) → `toytrain_pair04.png`

### valve

- pair 0: patch_time=[2,17) patch_freq=[31,37) → `valve_pair00.png`
- pair 1: patch_time=[13,28) patch_freq=[99,105) → `valve_pair01.png`
- pair 2: patch_time=[235,250) patch_freq=[72,78) → `valve_pair02.png`
- pair 3: patch_time=[253,268) patch_freq=[78,84) → `valve_pair03.png`
- pair 4: patch_time=[24,39) patch_freq=[40,46) → `valve_pair04.png`

## 4. Visual Assessment

### Patch-Boundary Artifacts

The replaced patch edges are visible only when donor and anchor
spectrograms differ substantially in energy level at the patch boundary.
When both samples have similar acoustic content, the transition is smooth.
The edge and enhanced views amplify patch-boundary contrast.

### Machine-Level Structure Preservation

Most pseudo-anomalies retain the overall machine-specific spectrogram
structure. The replaced patch introduces a local 'foreign' insertion
without destroying the broader machine-type acoustic identity.

### Individual machine observations:

- **bearing**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/bearing/`
- **fan**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/fan/`
- **gearbox**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/gearbox/`
- **slider**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/slider/`
- **toycar**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/toycar/`
- **toytrain**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/toytrain/`
- **valve**: 5 pairs saved to `results/v58a_pseudo_anomaly_visual_sanity/valve/`

## 5. Final Verdict

**V58A pass: eligible for V58 single-seed pilot human review.**

The same-machine TF patch replacement generates structurally coherent
pseudo-anomalies. Patch boundaries are minimally visible in logmel view.
The task appears learnable without trivial boundary-detection shortcuts.

---

No model training. No anomaly labels. No scoring. Fixed patch parameters.
