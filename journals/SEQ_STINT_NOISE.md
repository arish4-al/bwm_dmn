# Sequence vs stim/integ neurons: noise-matched control

Started 2026-11-03. Builds on [FIG5_CLUSTERS.md](FIG5_CLUSTERS.md), which records the cell selection.

## Question

In Figure 5, sequence neurons and stim/integ neurons look different in two places:

- within trial types: the bin x bin correlation (panels d/e)
- across trial types: the concatenated bin x bin correlation (h/j), the overall r between trial types, and the 2D cross-correlograms

Sequence neurons are much noisier than stim/integ neurons; they barely reproduce between the train and test halves. Are the correlation differences only a consequence of that noise? Test: add independent noise to the stim/integ neurons until they are as unreliable as the sequence neurons, then redo the analyses.

## Setup

- Script: `seq_noise_match.py` (new dataset by default; `--data old` for the cached stack, not run yet).
- The first run caches every computed result in `cache.pkl` in the output folder (about 150 MB). Later runs only redraw from that cache: all figures by default, or just the ones named with `--figs` (e.g. `--figs metrics fig5f`). A single figure takes about 10 s, against about 75 s for the full analysis. The cache is rebuilt if `--seeds` or the noise model changes, if `CACHE_VERSION` is bumped, or with `--recompute`.
- Data: the new stack, `concat_cvTrue.npy`. It has 12,182 sequence cells and 3,632 stim/integ cells, using the cluster picks in `fig5.py`.
- Analysed columns: the six stimulus-aligned trial types in `CONDITIONS` (432 bins), test half (`concat_z`, even trials), as in `fig5.py`.
- Outputs: `dmn/figs/figure5_new/noise_match/`. That folder holds the figures and `summary.json`, which has every number below.

### Noise statistics of the two groups

Split-half reliability is the per-cell Pearson r between the train and test halves over the 432 analysed bins.

| | median | mean |
|---|---|---|
| sequence | 0.016 | 0.044 |
| stim/integ | 0.396 | 0.408 |

On these six trial types, sequence cells are close to pure noise in the test half. Over all 1,872 bins their median is 0.077.

Noise is estimated as train minus test, which cancels the shared signal. Its autocorrelation within a trial-type segment is strongly smoothed in both groups:

| lag (bins) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| sequence | 1 | 0.83 | 0.66 | 0.50 | 0.35 | 0.19 | 0.04 |
| stim/integ | 1 | 0.81 | 0.63 | 0.47 | 0.31 | 0.15 | -0.01 |
| synthetic | 1 | 0.82 | 0.65 | 0.49 | 0.33 | 0.18 | 0.04 |

The sequence noise also has a small slow tail, about 0.03 at lags 6-10, that decays to 0 by about lag 30. White noise would therefore be the wrong model: the noise has to be coloured.

### Noise model

The stack only has trial-averaged PETHs per half, not single trials. Independent trial-by-trial noise is therefore modelled as its average over the trials in a half: Gaussian noise, independent per cell, per half and per trial-type segment. Its temporal autocorrelation is copied from the sequence cells' train-minus-test noise up to lag 40, using a zero-phase FIR kernel (`noise_kernel`). Because it is independent across cells, it adds no shared population structure.

**Shared noise between change_b and mistake** (added later; now the default, `SHARED_NOISE`): 25% of block_change_s trials are also mistake_s trials (see "Trial membership and counts"), so real trial noise is partly shared between those two segments. The added noise mixes the change_b noise into the mistake segment at the same bins, so that the two correlate at r = 0.067 in each half. This value is the median over sessions of k / (2·√(n_change · n_mistake)), with k the shared trials, assuming a shared trial lands in the same half of both types half the time (see the half-assignment evidence there). No other pair of the six types shares trials. `--independent-noise` reproduces the earlier fully independent model in `noise_match_independent/`.

For each stim/integ cell with shared variance s² and per-half variance σ² on the analysed bins, the added noise variance is v = s²/ρ_target − σ². This lowers the expected reliability to ρ_target. The noise is added to the whole row of both halves, and each row is re-z-scored as in the stack.

Noise levels (measured reliability after noise is for seed 0):

| variant | target | measured mean / median after noise |
|---|---|---|
| mean-matched (main) | uniform ρ = 0.054, calibrated so the mean measured reliability equals the sequence mean | 0.045 / 0.041 |
| rank-matched | each cell gets the sequence reliability at its rank, floored at 0.01 | 0.079 / 0.056 |
| ρ = 0.2 | uniform | 0.161 / 0.172 |
| ρ = 0.1 | uniform | 0.085 / 0.083 |

The rank-matched variant comes out less noisy than the sequence cells, because the measured sequence distribution already contains sampling spread and the floor removes its negative half. The mean-matched variant is the fair comparison.

Each variant was run with 5 noise seeds. The spread across seeds is at most 0.002 for every metric except `sim_to_seq` (about 0.015).

To separate real differences from cell-count effects, sequence cells were subsampled to 3,632 cells per seed (the "sequence (n matched)" row). The first version also drew a second, non-overlapping subsample as the reference for `sim_to_seq`. That metric and `overall_off` have since been removed from the script and `metrics.png`. Their values from the first run are kept in the table below; the overall r is now analysed in its own section.

## Results

Metrics are computed on the test half over the six trial types. "Local" means a lag under 6 bins, where the noise autocorrelation dominates.

The numbers in this section are from the fully independent noise model. With the shared change_b/mistake noise, every metric changes by at most 0.003: for example, the different-type same-time-bin r is 0.046 instead of 0.043, and `spread_within` is 0.017 instead of 0.018. The overall-r section below gives both models.

| metric | sequence | seq (n matched) | stim/integ | stim/integ + noise (mean-matched) |
|---|---|---|---|---|
| within type, mean r at lags 1-5 | 0.552 | 0.552 | 0.663 | 0.523 |
| within type, mean r at lags ≥ 6 | 0.028 | 0.027 | -0.062 | 0.002 |
| trial-type average, mean r at lags ≥ 6 | -0.034 | -0.036 | -0.156 | -0.045 |
| `spread_within`: growth of the within-type band (see below) | 0.001 | 0.003 ± 0.003 | 0.197 | 0.018 ± 0.002 |
| across types: different types, mean r | 0.000 | 0.000 | 0.020 | 0.001 |
| across types: different types, same time bin | 0.034 | 0.036 | 0.385 | 0.043 |
| `spread_cross`: growth of the cross-type band (see below) | -0.003 | -0.003 ± 0.000 | 0.141 | 0.014 ± 0.001 |
| overall r between trial types (off-diagonal; removed from `metrics.png`, see below) | 0.035 | 0.036 | 0.422 | 0.047 |
| `sim_to_seq` (pattern r with a separate sequence subsample, local band excluded; removed) | n/a | 0.777 (ceiling) | 0.565 | 0.568 |

The ρ = 0.2, ρ = 0.1 and rank-matched rows are in `summary.json` and `metrics.png`. They fall between the original and the mean-matched stim/integ rows.

A figure caption for `metrics.png` (groups, noise levels and each panel's definition) is written to `metrics_caption.md` in the output folder. The script regenerates it with the current numbers on every run. `metrics_captioned.pdf` is a one-page LaTeX build (pdflatex) with the vector `metrics.pdf` and the caption below it; its `.tex` source is kept beside it.

### Within trial types (panels d/e): explained by noise

`within_type_corr.png` and `lag_profiles.png`: with matched noise, the stim/integ bin x bin matrices look like the sequence ones. Both show a narrow diagonal band and almost nothing off the band. The lag profiles overlap almost exactly for the trial-type average, and nearly so per trial type.

The broad band and the negative far-off-diagonal lobes of the original stim/integ matrices are lost. The narrow band in the sequence matrices is therefore just the autocorrelation of their noise, not a sign of sequential structure.

One small difference remains. Sequence cells keep a weak positive r at long lags within a trial type (0.028 vs 0.002).

### Across trial types (panels h/j, overall r): magnitude explained, structure not

- Magnitude: with matched noise, the same-time correlation between trial types falls from 0.385 to 0.043, close to the sequence value of 0.034. The overall r between trial types falls from 0.422 to 0.047, against 0.035 for sequence cells. Most of the panel h-vs-j magnitude gap is noise.
- Structure: `cross_type_corr.png` still shows two qualitative differences.
  1. Noisy stim/integ blocks between different trial types keep a clean diagonal ridge (the same time bins correlate across trial types). In the sequence matrix that ridge is faint and patchy.
  2. The sequence matrix has warm same-trial-type blocks: r is positive across the whole block, most clearly for L_sL_cL_b and R_sR_cR_b. Same-type r over bin pairs at least 6 bins apart is 0.044 vs 0.013 (this metric was later dropped from `metrics.png` because it duplicates `far` with different weighting). A likely cause is a trial-type-specific offset per cell (a cell's level differs between trial types). Not yet checked.
- Adding noise does not make the stim/integ pattern more sequence-like. `sim_to_seq` (now removed) stayed at 0.57 (0.565 before, 0.568 after), well below the 0.78 that two sequence subsamples reach.

### Response spread over time within the trial type: still different

In the stim/integ matrices, the diagonal band widens later in the trial type: late responses are broader and overlap more in time. The sequence band keeps the same width throughout. `spread_over_time.png` and two metrics (`midlag_by_time`, `growth`) measure this.

- **Mid-lag r by time, m(t):** for bin t of a trial type, the mean r between t and every bin 6-15 bins away on either side.
  - Within type, this comes from that trial type's own matrix.
  - Across types, it comes from the blocks pairing that trial type with each of the other five, averaged over the five. So it is the width of the cross-type ridge at time t.
  - Lags under 6 are left out because there the noise smoothness dominates. Near the edges of the trial type, fewer bins are available, which causes small upturns in the first and last few bins.
- **`spread_within` / `spread_cross`:** the mean of m(t) over the last third of the trial type (bins 48-71) minus the first third (bins 0-23), averaged over the six types. Positive means the band widens over the trial.

| | sequence | seq (n matched) | stim/integ | stim/integ + noise (mean-matched) |
|---|---|---|---|---|
| `spread_within` | 0.001 | 0.003 ± 0.003 | 0.197 | 0.018 ± 0.002 |
| `spread_cross` | -0.003 | -0.003 ± 0.000 | 0.141 | 0.014 ± 0.001 |

(± is the SD over 5 seeds.)

- Stim/integ m(t) rises steadily across the trial in every trial type, both within and across types. Sequence m(t) is flat.
- With matched noise the stim/integ growth shrinks about 10x, but it stays positive and clearly above sequence cells. This holds in both measures and in most individual trial types; in the across-types measure, m(t) roughly doubles from early to late.
- The size of the shrinkage is about what noise alone should do. Added noise scales every r by about the signal fraction (about 0.05 / 0.4), and 0.197 × 0.11 ≈ 0.022 is close to the 0.018 measured.
- So the widening over time is a real property of stim/integ cells that survives sequence-level noise. Sequence cells don't show it.
- Sequence cells have a higher but flat within-type level in L_sL_cL_b and R_sR_cR_b, about 0.13. This is the warm-block effect above, not a widening.

### Overall r between trial types: what the structure is

Figure: `overall_structure.png`. Code: `overall_decomposition` and `plot_overall_structure`. Matrices and summaries are under `overall` in `summary.json`. The noisy matrices are averaged over the 5 seeds.

The overall r for a pair of trial types is the Pearson r between the two whole segments (cells × 72 bins, flattened), from `overall_corr_matrix`; the decomposition matches it to 1e-7. Each cell's segment is its mean level in that trial type plus a time course around that level. The time course averages to zero within each cell, so the covariance splits exactly into two parts:

- **level part:** r across cells between the per-cell mean levels in the two trial types
- **time-course part:** r between the segments after each cell's mean level is removed
- **level share:** the level covariance divided by the total covariance

Each panel in the figure has its own colour scale. "Pattern r vs sequence" is the r over the 15 trial-type pairs between a group's matrix and the sequence matrix.

| | overall r (mean off-diag.) | level part | time-course part | level share of covariance |
|---|---|---|---|---|
| sequence | 0.035 | -0.005 | 0.041 | -0.05 |
| stim/integ | 0.422 | 0.175 | 0.447 | 0.04 |
| stim/integ + noise, independent | 0.047 | 0.008 | 0.051 | 0.01 |
| stim/integ + noise, shared change_b/mistake noise (default) | 0.049 | 0.010 | 0.054 | 0.02 |

Pattern r with sequence cells over the 15 pairs:

| | overall | level | time course |
|---|---|---|---|
| stim/integ | 0.79 | 0.70 | 0.78 |
| stim/integ + noise, independent | 0.87-0.88 | 0.59-0.69 | 0.90-0.91 |
| stim/integ + noise, shared change_b/mistake noise | 0.84 | 0.69 | 0.83 |

The independent row gives two runs with different noise draws: the separate `--independent-noise` run, and the row in `overall_structure.png`. The level-part pattern r changes by 0.1 between them. Over only 15 small values it isn't stable, so read it loosely.

`overall_corr.png` shows the overall r only, one panel per group (the first column of `overall_structure.png`). `overall_structure.png` (default run) has four rows: sequence, stim/integ, noise-matched with independent noise, and noise-matched with shared change/mistake noise. In the shared row, change_b–mistake is the top pair in both the overall and time-course panels (0.10), and it also shows in the level part (0.04, vs 0.03 for sequence cells).

change_b–mistake overall r: sequence 0.076; stim/integ 0.409; noisy stim/integ 0.053 with independent noise and 0.096 with shared noise. Mean of the other 14 pairs: 0.032, 0.423 and 0.046.

Pure-noise baseline: coloured noise with the sequence cells' per-type SD, z-scored over all 21 types in the same way. It gives level r of about -0.06 for every pair, because z-scoring makes each row's segment means sum to about zero. The time-course r is 0 ± 0.004.

#### What the structure is

1. **In all three groups the overall r is carried by the time course, not by per-cell levels.** The level share is at most 5%. So the similarity between trial types comes from the same cells being active at the same times, not from cells having a constant offset across types.
2. **The pattern across trial-type pairs is largely shared, and mostly set by how noisy each trial type is.**
   - In every group the strongest pair is the two congruent types, L_sL_cL_b–R_sR_cR_b, followed by their pairs with mistake. The weakest pairs involve the incongruent types L_sL_cR_b and R_sR_cL_b.
   - Under independent noise, an observed r is the true r scaled by √(rel_a · rel_b), where rel is each type's reliability between halves (cells × bins). Per-type reliability:

     | | change_b | L_sL_cL_b | L_sL_cR_b | R_sR_cR_b | R_sR_cL_b | mistake |
     |---|---|---|---|---|---|---|
     | sequence | 0.026 | 0.151 | 0.035 | 0.135 | 0.038 | 0.060 |
     | stim/integ | 0.33 | 0.71 | 0.34 | 0.69 | 0.33 | 0.52 |

   - How well √(rel_a · rel_b) predicts the overall-r pattern:

     | | r with the prediction |
     |---|---|
     | stim/integ | 0.99 |
     | stim/integ + noise | 0.80 |
     | sequence | 0.68 |

   - So for stim/integ cells the matrix is essentially one common response, the same in every trial type: after correcting for noise, r for the congruent pair ≈ 0.61 / √(0.71 × 0.69) ≈ 0.88. Each trial type's noise sets how much of it shows.
   - Noise-matched stim/integ cells keep that pattern, which is why they match sequence cells so well in the time-course part (pattern r 0.91).
3. **Where sequence cells depart from that model:**
   - **change_b–mistake** is sequence cells' strongest pair (0.08). Given how unreliable these two types are, the most independent noise would allow is about √(0.026 × 0.060) ≈ 0.04. So the two segments must share noise. Confirmed: change_b and mistake share trials, and they are the only pair of the six that do (see "Trial membership and counts" below). The shared trials predict a noise r of about 0.067, which fits. The same shared noise also adds about 0.05 to the stim/integ change_b–mistake value (0.41 observed, against 0.36 predicted from reliability alone). It doesn't stand out there because the stim/integ segments are only 50-67% noise and every other pair is 0.27-0.61.
   - With the shared noise added to the noise model, change_b–mistake becomes the strongest pair in the noise-matched stim/integ cells too (0.096 against 0.046 for the other pairs, vs 0.076 against 0.032 in sequence cells). It slightly overshoots because the noisy copy keeps some real stim/integ signal for that pair (about 0.05). The pattern r with sequence cells drops from 0.91 to 0.83 (time-course part) because of that overshoot on a single pair; the rest of the matrix is unchanged.
   - **Level part:** against the about -0.06 baseline, sequence cells show positive level coupling of R_sR_cR_b with L_sL_cR_b and with L_sL_cL_b (+0.07 and +0.06). The pairs at about -0.06 (L_sL_cR_b–R_sR_cL_b, and each incongruent type with mistake) are at the noise baseline. Stim/integ levels are positive for every pair (0.10-0.27): cells that respond, respond in every type. In the noise-matched copy they shrink to between -0.03 and +0.03. The level part is where the groups differ most (pattern r 0.59-0.70), but it carries at most 5% of the covariance.
4. **Summary:** at matched noise, the overall-r matrices of stim/integ and sequence cells have nearly the same structure: a time-course similarity across types, scaled down by each type's noise. The extra change_b–mistake coupling in sequence cells comes from trials shared by the two types; noise-matched stim/integ cells show it too once the noise model shares those trials. What remains specific to sequence cells is a small level structure.

### 2D cross-correlograms (ctxt seq similarity): clearly still different

In `crosscorr2d_{seq,stint,stint_noisy}.png`, noisy stim/integ cells keep the smooth diagonal ridges of the original between every pair of trial types. These are latency-graded responses that are consistent across trial types, so averaging over 3,600 cells recovers them despite the per-cell noise. Sequence cells show only streaky texture, with no consistent ridge.

This is the strongest remaining difference. Caveat: the correlograms depend on cell order. Both groups are in the original Rastermap order, which was fit on the train half without noise.

### Population-average PETH (panel f)

`fig5f_noise.{png,pdf}` redraws fig5 panel f on the test half. It shows stim/integ, the mean-matched noisy copies with and without shared change/mistake noise (seed 0), and sequence cells.

- The noisy copies keep the slow ramp within each type, at about 40% of the original height: about 0.17 against 0.43 at the peak of the type average. Adding noise and then re-z-scoring each row scales the mean down.
- The shared and independent versions are indistinguishable. Shared noise changes only the change_b–mistake covariance, not the mean.
- Sequence cells stay flat near 0, so even with matched noise, panel f still separates the groups.
- The early stim transient (bins 5–12) disappears from the noisy copies. It is carried by the most reliable third of stim/integ cells: bump +0.26 at reliability 0.74. The low and middle thirds have the opposite sign, -0.07 and -0.15. With a uniform reliability target, the high-reliability cells get the most noise and shrink the most after re-z-scoring, about √(0.055/0.74) ≈ 0.27×. Their transient therefore no longer outweighs the others.

## Conclusions so far

1. Within-trial-type correlation: matched noise turns stim/integ cells into sequence-like matrices. Panels d vs e do not by themselves show that sequence cells differ from stim/integ cells beyond noise level.
2. Cross-trial-type magnitude: mostly explained by noise.
3. Cross-trial-type structure and response spread: not explained by noise. The band widens over the trial in stim/integ cells (within and across types) and survives matched noise; it is flat in sequence cells. Stim/integ cells share time-locked structure across trial types, which survives sequence-level noise in the bin x bin matrices and especially in the 2D cross-correlograms. Sequence cells lack it, and show same-trial-type block structure that noisy stim/integ cells lack.

## Reliability of sequence cells (moved)

Moved 2026-11-19 to [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md): split-half reliability with the train-half peak removed, where the peaks fall (the three noisy trial types), and split-half r on subsets of trial types.


## Trial membership and counts

Trial types are defined in `concat_PETHs` in `dmn_new.py` (the `tts` dict, `vers='concat'`):

| trial type | trials included |
|---|---|
| block_change_s | the first three trials after any block switch (`first_three_after_block_switch`), any stimulus, choice or outcome |
| stimLbLcL, stimLbRcL, stimRbRcR, stimRbLcR | stimulus side × block × choice, only trials at least 3 trials into the block (`deep_in_block`) |
| mistake_s | every error trial (`feedbackType == -1`) |

The overlaps were counted from the 417 cached trials tables under `Downloads/ONE/openalyx...`, with the same definitions. In place of the brainwidemap QC mask, a simplified mask was used: finite stimOn, firstMovement and feedback times, and a nonzero choice. These sessions are not necessarily exactly the ones in the stack. Script: the session scratchpad's `overlap.py`; not kept in the repo.

| | change_b | L_sL_cL_b | L_sL_cR_b | R_sR_cR_b | R_sR_cL_b | mistake |
|---|---|---|---|---|---|---|
| median trials per session | 30 | 168 | 30 | 156 | 33 | 107 |
| total trials | 13,997 | 77,391 | 14,116 | 71,558 | 14,358 | 46,848 |

- **Overlap:** 25.1% of block_change_s trials are also mistake_s trials (7.5% of mistakes). The four stimulus types contain only correct trials (0 errors), so they share no trials with mistake_s. `deep_in_block` keeps them out of the first three trials after a switch, so they share none with block_change_s either. change_b–mistake is the only overlapping pair of the six.
- **Predicted shared noise:** with about 7.5 shared trials per session out of 30 and 107, the noise in the two half-averages correlates at about 7.5 / √(30 × 107) ≈ 0.13 if the shared trials always land in the same half, or about 0.07 if only half of them do. Sequence cells' segments are almost all noise, so this predicts an overall r of about 0.07-0.13 for change_b–mistake. Observed: 0.08, all of it in the time-course part.
- **How trials are split into halves (indirect evidence):** in the train-minus-test noise, change_b and mistake are not correlated (r = -0.016, at the -0.006 to -0.014 of unrelated pairs). mistake_s and mistake_m, which use identical trial lists, are (r = +0.059).
  - If halves were assigned by global trial number, every shared trial would sit in the same half of both types, and change_b–mistake would correlate in train minus test as well.
  - If halves are assigned by position (odd/even) within each type's own list, identical lists share halves, while a change_b/mistake shared trial is in the same half only half the time. Its same-half and cross-half contributions then cancel in train minus test.
  - So the data fit the per-list split, giving a test-half shared-noise r of k / (2·√(n_change · n_mistake)): median 0.067 (interquartile 0.053-0.083) over sessions. This is the value used in the noise model.
- **Trial counts:** the three noisy types have 5.1-5.6x fewer trials than the congruent types. mistake has 107 per session, which fits its intermediate SD (0.90).

## Bin size and smoothing

In `dmn_new.py`, `T_BIN = 0.0125` s and `sts = 0.002` s. `_bin_peth` counts spikes in 12.5 ms windows at int(0.0125 // 0.002) = 6 offsets and interleaves them. So:

- each bin is about 2.08 ms;
- a 150 ms window ([0, 0.15] s after stimulus onset) gives 12 × 6 = 72 bins;
- each value is a 12.5 ms boxcar count, so neighbouring bins share spikes up to 5 bins apart.

This is the noise autocorrelation that falls linearly to about 0 at lag 6, and the ±6-bin steps in the peak-aligned average. In time units:

| quantity in bins | in ms |
|---|---|
| `LOCAL` = 6 | 12.5 |
| `MID_LAGS` 6-15 | 12.5-31 |
| each trial-type segment, 72 | the first 150 ms after the stimulus (or before the movement, for the `_m` types) |

## Firing-rate matched control

Added 2026-11-04. Script: `seq_fr_control.py` (new stack only). Outputs: `dmn/figs/figure5_new/fr_control/`, which holds `fr_control.png`, `fr_control_rasters.png`, `alignment.png` and `summary.json` with every number below. One full run takes a few minutes; most of that is the Rastermap fit on the partners.

Why: sequence cells fire much less than other cells. `fr` is the mean of the raw PETH, which is probably the trial-averaged count per 12.5 ms window (so 0.029 would be about 2.3 Hz; not checked).

| | median `fr` |
|---|---|
| sequence | 0.029 |
| stim/integ | 0.125 |
| all other cells | 0.083 |

Sequence cells are 39-44% of the lowest-rate quarter of all cells (fr quantile bins 1-3 of 12), and 3% of the highest bin. With rows z-scored, a few coincident spikes make a large peak in a trial type with about 15 trials per half. Are sequence cells different from other cells at the same rate?

### Partner sets

Each sequence cell gets one partner. Partners are drawn from the remaining cells (not sequence, not stim/integ), by greedy nearest-neighbour matching without replacement:

| partner set | matched on | match quality |
|---|---|---|
| fr+region | log fr, within the same Beryl region | median fr 0.032; KS 0.04; 99.96% in the same region; 0.7% reused |
| fr | log fr only | KS 0.001 |
| fr+peak | same train-peak trial type (of the six), nearest in log fr and train peak height | height 5.6 vs 6.5 z; median fr 0.039; KS 0.11 |
| fr+peak+rel15 (scratchpad check, not in the script) | fr+peak plus split-half r over the 15 quiet types | rel15 0.248 vs 0.154; height 5.5 vs 6.5 |

Sequence cells sit in a corner of the space: the lowest rate, the tallest train peaks, the lowest reliability. The pool doesn't fully reach that corner, so the last two matches stay imperfect in the expected direction.

About 20% of low-rate cells have no spikes at all in a noisy segment in one half (14-17% of sequence cells, 20-22% of their fr+region partners, 1% of stim/integ cells). Such a segment is constant, and its argmax is meaningless. The per-segment metrics below therefore leave those cells out: height metrics use cells with spikes in the train half, and peak shift and r use cells with spikes in both.

### Results

Split-half r, median (mean):

| | six | noisy 3 | quiet 3 | all 21 | 15 quiet types |
|---|---|---|---|---|---|
| sequence | 0.016 (0.044) | 0.000 (0.032) | 0.063 (0.089) | 0.077 (0.103) | 0.154 (0.191) |
| fr+region | 0.061 (0.105) | 0.018 (0.063) | 0.120 (0.165) | 0.223 (0.281) | 0.351 (0.385) |
| fr | 0.072 (0.120) | 0.016 (0.068) | 0.138 (0.183) | 0.256 (0.315) | 0.390 (0.416) |
| fr+peak | 0.078 (0.117) | 0.041 (0.084) | 0.149 (0.192) | 0.214 (0.252) | 0.353 (0.376) |
| fr+peak+rel15 | 0.048 | | | | |
| stim/integ | 0.396 (0.408) | 0.314 (0.337) | 0.614 (0.555) | 0.482 (0.483) | 0.662 (0.619) |
| all other | 0.123 (0.175) | 0.081 (0.120) | 0.215 (0.266) | 0.362 (0.399) | 0.535 (0.517) |

Peaks over the six types:

| | train peak in noisy types | test peak in same type (chance) | peak-time r in own type | median shift | z at train peak, train / test |
|---|---|---|---|---|---|
| sequence | 0.94 | 0.35 (0.28) | 0.11 | 20 | 6.46 / 0.16 |
| fr+region | 0.79 | 0.33 (0.24) | 0.22 | 18 | 4.68 / 0.31 |
| fr+peak | 0.94 | 0.38 (0.28) | 0.20 | 18 | 5.61 / 0.39 |
| fr+peak+rel15 | 0.94 | 0.36 | 0.12 | 20 | 5.51 / 0.23 |
| stim/integ | 0.88 | 0.33 (0.27) | 0.57 | 8 | 4.95 / 1.35 |

What these show:

- **Firing rate does not explain the low reliability.** At every firing rate, sequence cells are less reliable than other cells (`fr_control.png`, middle panels). On the 15 quiet types, the gap is 0.12 vs 0.29 in the lowest-rate bins and 0.31 vs 0.83 in the highest. Every partner set is 2-4x more reliable than the sequence cells, on every subset of types, including the 15 quiet types where sequence cells were not selected.
- **The train-peak carry-over is not larger than in matched cells.** (2026-11-05 correction: this section originally said "no sign of a sequence". That overstated it. The faint test-half diagonal is real signal, and the cross-half analysis in [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md) shows it tiles time. What this control shows is only that the train-peak measure doesn't single out sequence cells.) Partners that pass the same peak selection (same peak type, similar height) carry their peaks over at least as well as sequence cells: test height at the train peak 0.39 vs 0.16, peak-time r 0.20 vs 0.11. With off-peak reliability also matched, the partners still match or beat sequence cells (0.23 vs 0.16; r 0.12 vs 0.11).
  - So the faint test-half bump (0.16 z) is no larger than what any low-rate cell selected on its train peak shows.
  - It is a little smaller, because sequence cells have the least real signal of all.
- **Rastermap on the partners finds no diagonals.** The fr+region partners, given their own Rastermap fit on the train half, sort into blocky responses that reproduce in the test half (`fr_control_rasters.png`, column 2). There is no diagonal.
  - Sorting either set by train peak time gives an equally sharp diagonal that is mostly gone in the test half (columns 3-5).
  - In those sorts, the change_b rows also light up in block_change_m in both sets. That is the same trials counted in both windows, not a property of sequence cells.
- **Reading:** the sharp train-half diagonals that defined the clusters are noise peaks in the fewest-trial types. Sequence cells are also the least reliable low-rate cells overall. Their weak but reproducible timing is analysed in [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md), "Cross-half bin x bin correlation". Why they are less reliable than other cells at the same rate (for example bursty or unstable units, or multi-unit contamination) is not known; unit QC metrics would show it.

### Stimulus- vs movement-aligned

Each of the six types has a stimulus-aligned copy ([0, 150] ms after the stimulus) and a movement-aligned copy ([-150, 0] ms before first movement), built from the same trial conditions. The movement copies of the four stimulus types skip `deep_in_block`, so they include more trials. For each segment and group the script measures:

- the split-half r of the time course
- the test/train height ratio at the train peak
- the median shift of the peak time between halves
- the test-half SD

It does this for all cells, and for each sequence family on its own diagonal type against the same cells' partners (`alignment.png`, `alignment` in `summary.json`).

- **Stim/integ cells are stimulus-locked.** Their reliability drops from 0.72 to 0.50 (L_sL_cL_b) and 0.34 to 0.17 (change_b) in the movement copy, and peak shifts grow from 5-9 to 8-14 bins. The movement copies are less noisy, so this is a real alignment preference.
- **Sequence cells prefer neither alignment.** In both copies of every type they are at or below the fr-matched partners: split-half r -0.03 to 0.08, peak-time r 0.08-0.23, median shift 17-19 bins against about 21 for chance.
- **The families on their own diagonal type:**

  | family | z at train peak, stim / move | test bump, stim / move (± SEM) | fr+peak partners' test bump, stim / move | train peak in the move copy |
  |---|---|---|---|---|
  | R_sR_cL_b | 6.31 / 4.34 | 0.14 ± 0.03 / 0.03 ± 0.03 | 0.30 / 0.13 | 17% |
  | L_sL_cR_b | 5.44 / 5.25 | 0.13 ± 0.03 / 0.10 ± 0.03 | 0.25 / 0.15 | 44% |
  | change_b | 5.35 / 5.20 | 0.08 ± 0.03 / 0.05 ± 0.02 | 0.23 / 0.11 | 44% |

  - The R_sR_cL_b family was selected in the stimulus copy (only 17% have their train peak in the movement copy).
  - The other two families split almost evenly. That fits clusters 45-47 carrying the L_sL_cR_b diagonal only in `sLbRchoiceL`.
  - In no family and in neither alignment does the test bump exceed the matched partners'.
- **Conclusion (single-cell peaks only):** carry-over of each cell's train peak is no larger than in matched cells, in either alignment. Per-cell peaks are too noisy to settle the question. The population cross-half analysis ([SEQ_RELIABILITY.md](SEQ_RELIABILITY.md)) does show a preference: sequence-cell timing is stimulus-locked.

## Cross-half bin x bin correlation (moved)

Moved 2026-11-19 to [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md): the train x test bin x bin correlation (sequence cells tile the 150 ms after the stimulus reproducibly, mostly in the congruent types), and how much of that tiling is shared across trial types.


## Spike-thinned stim/integ cells

Added 2026-11-13. Scripts are in `stint_thin/`. Status: run on ORCD 2026-11-13; metrics below (2026-11-14).

### Why the Gaussian noise control is not enough

Rastermap, and every figure 5 panel, sees only the z-scored rows (`concat_z_train` / `concat_z`). Z-scoring removes the rate scale, but a low rate still shows in the shape of a row. At about 2 Hz with about 15 trials per half, most bins are 0, and a few spikes give tall, sparse peaks. The mean-matched Gaussian noise is added directly to the z-scored rows (`add_noise`, then re-z-scored), and it pushes the rows the other way. Train half, six analysed types, per-cell medians (noise: ρ = 0.054, seed 0):

| | skew | kurtosis | max z | min z | fraction of bins at the row floor |
|---|---|---|---|---|---|
| sequence | 1.93 | 4.44 | 6.08 | −0.88 | 0.50 |
| stim/integ | 0.89 | 0.86 | 4.96 | −1.33 | 0.11 |
| stim/integ + Gaussian noise | 0.10 | 0.00 | 3.17 | −2.65 | 0.00 |

The noise matches reliability but not shape. It cannot produce the tall isolated peaks that the "diagonals are noise peaks in the fewest-trial types" reading depends on. Real count noise also scales with each bin's rate and with 1/n trials per type, while the Gaussian model uses one SD per row.

### What `fr` is

The stack's `fr` is the mean of the raw test-half PETH (`concat`, even trials, all 21 types, 1,872 bins) before z-scoring. The stack has no raw train half: `X_odd` and `X_even` are float32 copies of `concat_z_train` and `concat_z` (`pipeline/02_stacks.py`, `add_x_arrays`).

### Stack builder

The current builder is on `int-brain-lab/bwm_dmn` main, not in the local `dmn_new.py` (stale). It is `pipeline/02_stacks.py` → `dmn_bwm.stack_concat(cv=True)`. Its `_oddeven_means_concat` sends odd-numbered trials (1st, 3rd, …) to train and even-numbered trials to test, in session order within each trial type's own list. This settles the open item on how halves are assigned. The per-insertion bundles come from `dmn_bwm.concat_PETHs`:

- `load_good_units` for the spikes;
- `load_trials_and_mask` with 6 saturation intervals;
- 12.5 ms boxcar counts at a 2 ms stride.

### Rebuild from the prior-mech insertion cache

`manifold/insertion_cache/{eid}_{probe}.npy` from prior-mech-analysis holds spike times and cluster ids, and the full cache is on ORCD. Together with `bwm_tables/trials.pqt` (event times, `bwm_include`, saturation flags) and `bwm_tables/clusters.pqt` (uuid → `cluster_id`), it rebuilds the stack with no ONE or Alyx. On the 6 locally cached insertions that are in the stack (696 cells), the rebuilt even half equals `concat`, z(odd half) equals `concat_z_train`, and the rebuilt `fr` equals `fr`, all exactly (difference 0.0). One detail: spikes must be restricted to the requested clusters before `bin_spikes2D`, or it fails on insertions with dropped units.

### The job

- `prep_cells.py` (local, needs the stack) writes `data/stint_cells.npz` (336 KB, committed). It holds:
  - the 3,632 stim/integ cells in fig5 order, with pid, eid, probe, uuid and `cluster_id`;
  - the keep probabilities;
  - fingerprints of the stack rows, so ORCD can check the rebuild without the stack.

  They come from 436 of the 515 insertions. `data/insertions.csv` is upstream `pipeline/insertions.csv`.
- `stint_thin.py compute` handles one shard. For each insertion it loads the cache, keeps each spike of a stim/integ cell with probability p (one draw per seed, shared by both halves), rebins the 21 types and averages odd and even trials. It writes `shards/{pid}.npz`, including spike counts and trial counts per type and half.
- `stint_thin.py stack` checks that variant `p100` (no thinning) reproduces `fr`, Σ concat² and z(train)·v. It fails otherwise. It then writes `OUT/{variant}.npz` with P0 and P1 (seeds × cells × 1,872), in `stint_cells.npz` order.
- Variants:

  | variant | p |
  |---|---|
  | `p100` | 1 (check) |
  | `rate_rank` | each cell goes to the sequence fr at its fr rank: median p 0.233, range 0.11-1 |
  | `rate_median` | 0.233 for every cell (median seq fr 0.029 / stim/integ 0.125) |

  Seeds 0-4. Both thinned variants bring the median fr to 0.029, the sequence median.
- Local test (`SLURM_JOB_ID=test`, alyx-host `trials.pqt`, 6 insertions, 19 cells, 2 seeds):
  - `p100` passes, with max |fr| 7e-9;
  - the kept-spike ratio equals p per cell;
  - 4-8 s per insertion;
  - Alyx requests per job × jobs = 0, since the job never imports ONE.
- Run on ORCD (the agent does not submit). Pull the commit there first, then:

  ```bash
  bash stint_thin/submit_stint_thin.sh
  ```

  The defaults are 8 shards on `pi_fiete` at 6 G, then the stack job. Output goes to `$ONE_CACHE_DIR/dmn/res/stint_thin/`. `PARTITION=mit_preemptable` adds `--requeue`, and shards skip insertions already written.

### Results: noise-match metrics on thinned cells

`seq_thin_match.py` reads `OUT/{variant}.npz` (rsynced to `~/Downloads/stint_thin/`). It z-scores each half over all 1,872 bins, as the stack builder does, and runs the `seq_noise_match.py` metrics on the six analysed types of the test half. Sequence, stim/integ and Gaussian rows come from the `noise_match/cache.pkl`. Output is in `figs/figure5_new/thin_match/` (`summary.json`, `metrics.png`, `reliability.png`, `rasters_test.png`, `within_type_corr.png`, `cross_type_corr.png`).

- `p100` reproduces the stack: max |Δz| 6.5e-6 (float32), and every metric equals stim/integ.
- Thinning leaves 1–11 of 3,632 rows without a spike in one half, depending on the seed. Those rows are set to 0.
- `rate_rank` and `rate_median` agree to within seed SD on everything, so only rank is quoted below.

Reliability and row shape (test half, per-cell medians; ρ is the mean over cells and 5 seeds):

| | ρ mean | ρ median | skew | kurtosis | max z | min z | floor frac |
|---|---|---|---|---|---|---|---|
| sequence | 0.044 | 0.016 | 1.84 | 3.89 | 5.56 | −0.87 | 0.50 |
| stim/integ | 0.408 | 0.396 | 0.90 | 0.95 | 4.98 | −1.32 | 0.12 |
| stim/integ + Gaussian (mean) | 0.046 | 0.041 | 0.10 | 0.02 | 3.15 | −2.65 | 0.00 |
| stim/integ thinned (rank) | 0.176 | 0.133 | 1.66 | 3.08 | 5.92 | −0.82 | 0.45 |

Correlation metrics (mean ± SD over 5 seeds):

| | local | far | spread within | cc between, diag | spread cross | overall r full | level | time |
|---|---|---|---|---|---|---|---|---|
| sequence | 0.552 | 0.028 | 0.001 | 0.034 | −0.003 | 0.035 | −0.005 | 0.041 |
| stim/integ | 0.663 | −0.062 | 0.197 | 0.385 | 0.141 | 0.422 | 0.175 | 0.447 |
| + Gaussian (mean) | 0.523 | 0.002 | 0.017 | 0.046 | 0.014 | 0.049 | 0.010 | 0.054 |
| + Gaussian (rank) | 0.535 | −0.008 | 0.028 | 0.083 | 0.020 | 0.087 | 0.039 | 0.092 |
| thinned (rank) | 0.581 | −0.037 | 0.080 ± 0.003 | 0.187 ± 0.002 | 0.051 ± 0.002 | 0.203 | 0.075 | 0.215 |

Readout:

- **Row shape: rate explains it.** Thinning to the sequence rates gives sequence-like rows: skewed, heavy-tailed, about half the bins at the floor, and tall peaks. The Gaussian control gives the opposite. So rate alone accounts for the shape difference that Rastermap sees.
- **Reliability: rate explains about two-thirds of the gap, not all of it.** At sequence-matched rates, stim/integ ρ falls from 0.41 to 0.18, but sequence cells are at 0.04. Sequence cells are less reliable than stim/integ cells at the same rate. This agrees with the firing-rate matched control.
- **Structure: thinned stim/integ cells keep about half of it.** Every structure metric lands about halfway between stim/integ and sequence: cross-type diagonal 0.19, spread within 0.08, overall r 0.20. That is far above sequence, and above the Gaussian controls, which were matched to the lower sequence reliability. So rate alone does not make stim/integ cells look like sequence cells. The comparison is not yet at equal reliability, though.

### Thinned + noise: rate and reliability both matched

`seq_thin_match.py --noise` adds noise to the thinned rows until the mean split-half reliability equals the sequence mean (0.044). The level is calibrated on seed 0 and then applied to 5 fresh noise draws per seed. Both noise models use the sequence-noise temporal kernel and the shared change/mistake noise.

- **gauss**: `add_noise` on the z-scored rows, the same model as the Gaussian control. Uniform target ρ is 0.084 (rank) / 0.085 (median).
- **count**: noise on the raw half-average rate, with SD c·√P per bin, then z-scored. c = 0.52 (rank) / 0.51 (median). Bins with no spikes get no noise. Nonzero bins can go negative, though, so the floor is no longer at the zero bins.

Flat thinned rows get no noise and are set to 0. Rank and median again agree, so rank is shown:

| | ρ mean | skew | kurtosis | max z | min z | local | far | spread within | cc diag | spread cross | overall r | level | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sequence | 0.044 | 1.84 | 3.89 | 5.56 | −0.87 | 0.552 | 0.028 | 0.001 | 0.034 | −0.003 | 0.035 | −0.005 | 0.041 |
| sequence (n matched) | | | | | | 0.552 | 0.027 | 0.003 ± 0.003 | 0.036 | −0.003 | 0.036 | −0.004 | 0.042 |
| stim/integ + Gaussian (mean) | 0.046 | 0.10 | 0.02 | 3.15 | −2.65 | 0.523 | 0.002 | 0.017 | 0.046 | 0.014 | 0.049 | 0.010 | 0.054 |
| thinned + gauss | 0.043 | 0.71 | 0.87 | 4.36 | −2.23 | 0.525 | −0.009 | 0.027 ± 0.004 | 0.055 | 0.018 | 0.058 | −0.012 | 0.065 |
| thinned + count | 0.041 | 1.26 | 4.31 | 5.43 | −3.12 | 0.446 | −0.008 | 0.010 ± 0.003 | 0.044 | 0.006 ± 0.003 | 0.046 | 0.001 | 0.049 |

Readout:

- **With rate and reliability both matched, the stim/integ metrics fall to about sequence levels.** With count noise, cross-type diagonal is 0.044 vs 0.034, overall r 0.046 vs 0.035, the level part about 0, and spread cross 0.006 vs −0.003.
- **Spread within type**: thinned + count gives 0.010 ± 0.003, against 0.001 for sequence (0.003 ± 0.003 for the n-matched subsample), so the gap is about 2 SD of seed noise. The earlier "spread over time is still different" result (Gaussian control: 0.017) shrinks to this once the rate is matched as well. It still needs the deferred bootstrap over insertions.
- **Local r** is lower for count noise (0.45 vs 0.55). Rate-scaled noise is less smooth within a row than the sequence cells' own noise, so this metric reflects the noise model, not structure.
- **Thinned + gauss** sits slightly above the original Gaussian control on every structure metric. Its rows lose the sparse shape (floor 0, kurtosis 0.9), so it is the weaker of the two models.
- **Conclusion**: at the half-average level, a stim/integ population thinned to sequence rates and degraded to sequence reliability is close to indistinguishable from the sequence population on the figure 5 correlation metrics. What separates the groups in these metrics is mostly rate plus reliability. That leaves one question open: why are sequence cells less reliable than stim/integ cells at the same rate? See the unit-quality item below.

Figures: `thin_match/fig5f_thin_noise.png` (panel f) and `thin_match/metrics_thin_noise.png` (metric bars), rank variant only.

- **Panel f is the exception.** The population-average PETH of thinned + count still ramps up within each trial type, to about 0.1 z by mid-trial, while sequence stays flat near 0. Thinned + gauss and the original Gaussian control sit higher, at about 0.15–0.18. The noise shrinks the average response but cannot remove it, because the stim/integ cells share a response sign that sequence cells lack. The correlation metrics miss this, because they are computed across cells after each row is centred. So a shared mean response still separates the groups after rate and reliability are matched.

Next:

- 2D cross-correlograms (ctxt seq similarity) for thinned + count. They are not in `seq_thin_match.py` yet, and they were the clearest remaining difference under the Gaussian control.
- Fit Rastermap on the thinned (+ count) train half, to see whether train-only diagonals appear.

### Rastermap with the thinned + count cells added

Added 2026-11-16. Script: `seq_thin_rastermap.py` (`fit`, then `plot`). Output: `figs/figure5_new/thin_rastermap/` (`fit.npz`, `summary.json`, `rasters.png`, `composition.png`), with copies in `journals/figs/thin_rastermap_*.png`.

- Synthetic cells: `rate_rank` seed 0, count noise with c = 0.523, a fresh noise draw (seed 2026). Mean split-half ρ is 0.039, against 0.044 for sequence cells.
- Their train-half rows were appended to `concat_z_train` of the full stack, keeping all 54,569 real cells and their original stim/integ parents. Rastermap was refit with the fig5 settings on all 58,201 rows. One synthetic row has no train spike and was dropped. Real cells keep their group labels from the stored fit.
- Settings: `grid_upsample=0`, which is what the stored fit used. It makes `rm_labels` contiguous along `isort`, and the old `FIG5_CLUSTERS.md` snippet left it out (now fixed). The cluster assignment does not depend on it, only the order inside the sort does. With these settings, a refit on the real stack alone is deterministic (ARI 1.0 run to run), but it does not reproduce the stored fit (ARI 0.36). New cluster ids therefore do not map onto the fig5 picks.
- Whole-sort rasters, the same layout used to pick clusters: `rm_clusters_{0_24,25_49,50_74,75_99}.png`. A left strip marks each row's group, and each label gives the cluster id with its synthetic share.

| group | cells | in sequence-majority clusters | in stim/integ-majority clusters |
|---|---|---|---|
| sequence | 12,182 | 0.81 | 0.01 |
| stim/integ (real) | 3,632 | 0.06 | 0.78 |
| thinned + count | 3,631 | 0.45 | 0.21 |
| other | 38,755 | 0.07 | 0.03 |

"Sequence-majority" (28 of 100 clusters) means sequence cells make up more than half of the cluster's sequence + stim/integ + synthetic cells. "Stim/integ-majority" (9 clusters) is defined the same way. 14% of synthetic cells share a cluster with their own parent cell, against 1.9% for a random stim/integ cell.

- **No cluster of their own.** Synthetic cells are 6.2% of all cells. No cluster is more than 24% synthetic, and none is majority synthetic. The highest shares (14–24%) are in the stim/integ clusters 24–29 and 33. Next come the sequence-diagonal clusters (17–22, 30–32, 38–45, 55–69) at 10–15%, about twice the base rate. The non-task clusters (50–54, 70–95) are at 0–5%.
- **Rastermap sorts the synthetic cells into sequences.** The train raster (`rasters.png`, middle column) has sharp single-type diagonals, in block_change_s, stimLbRcL, stimRbRcR and stimRbLcR. They are absent from the test half, which is the signature used to pick the real sequence clusters.
- **They co-cluster with the sequence cells.** Across clusters, the synthetic distribution tracks the sequence distribution, not the stim/integ peaks (`composition.png`).
- **Some stim/integ structure survives.** About a fifth of the synthetic cells stay in stim/integ clusters. One block (rows about 1,000–1,500) keeps a short latency-graded response at each type onset that reproduces in the test half, which is the residual shared response seen in panel f.
- Caveat: one thinning seed and one noise draw. Synthetic cells were not re-picked by eye as sequence clusters, so it is untested whether a blind picker would count them.

## Significance of the spread difference

Added 2026-11-14. Script: `seq_spread_stats.py`, with the numpy-only worker code in `spread_core.py`. Output: `figs/figure5_new/thin_match/spread_stats.json`.

**The statistic.** D = spread(control) − spread(sequence), on the test half and the six analysed types. Spread is computed with the original thirds (bins 0–23 vs 48–71), and with trimmed thirds over bins 6–65 (6–25 vs 46–65), which keeps away from the trial-type edges. The faster spread code reproduces the old values: sequence 0.0015 / −0.0026, stim/integ 0.197 / 0.141.

**Bootstrap over insertions (paired), 1000 iterations.**
- Each iteration resamples the 515 insertions with replacement. A resampled insertion brings all of its sequence cells and all of its control cells.
- Noisy controls get a random draw from a bank of 20 precomputed noise draws. Thinned controls also get a random thinning seed.
- p is two-sided, and 0.001 is the floor ("≤.001" below).
- A second comparison uses sequence subsampled to the control's cell count (n matched). It changes D by less than 0.001 and gives similar p, so the difference in cell count does not matter.

**Implementation notes.**
- The first version regenerated noise in every iteration, at about 1.1 s per noisy control. It ran single-threaded at about 4 s per iteration. The bank plus 6 joblib workers brought the run down to 4 min for the bootstrap.
- Workers must not import `fig5`. That import builds a ONE client in `dmn_ari`, and six parallel workers race on `~/.one`.

**Within-insertion label permutation: discarded.** The shuffle keeps each group's cell counts per insertion. The control group therefore draws most of its cells from insertions rich in stim/integ cells, so its shuffled version still holds more stim/integ cells than the shuffled sequence group. The null mean of D comes out at 0.089 for stim/integ and 0.022 for thinned + count, not 0, so the null is not "no difference" and the noisy controls get negative z. For the partner sets the null is centred near 0 (0.003–0.011). There, the permutation agrees with the bootstrap at z of 5–14.

Bootstrap results: D vs sequence [95% CI], two-sided p.

| control | within, thirds | within, trimmed | cross, thirds | cross, trimmed |
|---|---|---|---|---|
| stim/integ | 0.196 [0.172, 0.216] ≤.001 | 0.161 ≤.001 | 0.144 ≤.001 | 0.114 ≤.001 |
| + Gaussian (mean) | 0.018 [0.005, 0.027] .002 | 0.017 [0.004, 0.027] .010 | 0.017 ≤.001 | 0.016 ≤.001 |
| + Gaussian (rank) | 0.026 ≤.001 | 0.022 .004 | 0.022 ≤.001 | 0.018 ≤.001 |
| + Gaussian (ρ = 0.1) | 0.031 ≤.001 | 0.028 ≤.001 | 0.026 ≤.001 | 0.024 ≤.001 |
| + Gaussian (ρ = 0.2) | 0.062 ≤.001 | 0.054 ≤.001 | 0.047 ≤.001 | 0.040 ≤.001 |
| thinned | 0.079 ≤.001 | 0.060 ≤.001 | 0.054 ≤.001 | 0.039 ≤.001 |
| thinned + gauss | 0.024 ≤.001 | 0.021 ≤.001 | 0.019 ≤.001 | 0.017 ≤.001 |
| thinned + count | 0.010 [0.001, 0.023] .032 | 0.008 [−0.002, 0.019] .118 | 0.008 [0.003, 0.015] ≤.001 | 0.007 [0.001, 0.014] .016 |
| fr+region partners | 0.039 [0.031, 0.047] ≤.001 | 0.031 ≤.001 | 0.013 ≤.001 | 0.010 ≤.001 |
| fr partners | 0.049 ≤.001 | 0.043 ≤.001 | 0.016 ≤.001 | 0.013 ≤.001 |
| fr+peak partners | 0.064 ≤.001 | 0.054 ≤.001 | 0.027 ≤.001 | 0.023 ≤.001 |

Bootstrap SD of the sequence spread itself is 0.003 within and 0.0016 cross.

Readout:

- **Sequence spread is about 0, and lower than every control.** That includes the real fr-matched cells, which have no added noise and the same rates: D 0.04–0.06 within and 0.01–0.03 cross, all p ≤ .001. So the lack of spread is a property of sequence cells, not of their low rate. Real cells of the same rate show it.
- **The thinned + count control is the only one close to sequence.**
  - Spread within: p = .032 with the original thirds. With the edges trimmed it is not significant (p = .118; n matched .19).
  - Spread cross: still significant (p ≤ .001 thirds, .016 trimmed; n matched .012 / .036), but the effect is small at 0.007–0.008.
  - So the synthetic control comes close to sequence cells when rate and reliability are both matched and the noise is rate-scaled. Real cells at the same rate do not.
- **Multiple comparisons:** 11 controls × 4 metrics is 44 tests, so the Bonferroni threshold is .0011. Every "≤.001" result survives it. The thinned + count results do not, nor do Gaussian (mean) within and Gaussian (rank) within, trimmed.
- **Caveats.**
  - Partner sets are one match (seed 0), not resampled.
  - The bank holds 20 noise draws, while the seed SD of spread is about 0.003, well below the bootstrap SD.
  - p is floored at .001 by 1000 iterations.

## Open items / next steps

- Block information in sequence cells, tested directly: blockL vs blockR segments (cross-half, as in [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md)), and the per-cell level coupling between trial types that share a block.
- Separate stimulus side from choice side in the side information. This needs error trials split by side, which the stack doesn't have (mistake_s pools both sides).

- Unit quality of sequence cells vs fr-matched cells: presence ratio, amplitude, drift, ISI violations. This would explain why sequence cells are less reliable than other cells at the same rate. It needs the clusters tables, not `brainwidemap`.
- Null for the sequence picks: take a stack of pure noise matched in per-type SD and smoothing, run the same Rastermap fit, and see whether diagonals of similar sharpness and cell count appear. Partly answered by the firing-rate matched control: fr-matched cells sorted by train peak give equally sharp diagonals. What is still open is whether the whole-stack Rastermap would cluster pure noise rows into diagonal clusters.
- ~~Find the code that built the odd/even stack.~~ Done 2026-11-13: upstream `dmn_bwm._oddeven_means_concat`, by position within each type's list (see "Spike-thinned stim/integ cells").
- Unit quality: `bwm_tables/clusters.pqt` has presence ratio, drift, contamination, amplitude and sliding-RP violations per uuid, so this needs no job.
- ~~Significance of the spread-over-time difference.~~ Done 2026-11-14 with an insertion bootstrap (see "Significance of the spread difference"). The within-insertion permutation was discarded because its null is biased.

- Check whether the sequence same-type block structure is per-cell trial-type offsets. Compare the variance of segment means between groups, and redo the across-type analysis after subtracting each cell's per-segment mean.
- Run `--data old` for the cached stack.
- The noise is modelled at the half-average level. If single-trial PETHs can be rebuilt (this needs `brainwidemap` and the raw insertions), add noise per trial and re-average, and check whether trial counts per trial type differ between the groups.
- Opposite control: average or denoise sequence cells, for example by fitting on train and testing on the projected test half, to see whether any shared cross-type structure appears.
- Remove the noise-induced diagonal band from the cross-correlograms before comparing them, for example by subtracting the cross-correlogram of the train-minus-test noise.
