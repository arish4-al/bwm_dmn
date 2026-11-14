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

## Reliability with the train-half diagonal removed

Run: `python seq_noise_match.py --peak-mask` (`run_peak_mask`). Outputs: `peak_masked_reliability.{png,json}` in the same folder.

Why: sequence cells were picked because their train-half peaks form a sharp diagonal, and that diagonal is mostly gone in the test half. The low reliability could therefore come from the peak alone, a large train-only deviation that dominates the correlation, while the rest of the row (the faint sequence) still reproduces.

Method:
- For each cell, find the peak bin of the train PETH over the six trial types.
- Drop a window of ±w bins around it, inside that trial type, from both halves.
- Correlate train and test over the remaining bins.
- Control: drop a window of the same size around a random bin instead.

### Results

The table gives the median split-half r, with the mean in brackets.

| half-width w (bins) | sequence, peak removed | sequence, random removed | stim/integ, peak removed |
|---|---|---|---|
| none (all bins) | 0.016 (0.044) | — | 0.396 (0.408) |
| 0 | 0.019 (0.045) | 0.018 | 0.399 |
| 2 | 0.023 (0.045) | 0.018 | 0.393 |
| 6 | 0.025 (0.046) | 0.018 | 0.393 |
| 10 | 0.025 (0.047) | 0.017 | 0.399 |
| 20 | 0.026 (0.048) | 0.016 | 0.410 |

#### Variant: remove each half's own peak

Here the window around the train peak and the window around the test peak are both dropped, from both halves (`both` in the json). The control drops two random windows (`rand2`).

| half-width w (bins) | sequence, both peaks removed | sequence, 2 random removed | stim/integ, both peaks removed |
|---|---|---|---|
| 0 | 0.020 (0.045) | 0.018 | 0.398 |
| 2 | 0.027 (0.047) | 0.019 | 0.391 |
| 6 | 0.033 (0.053) | 0.017 | 0.394 |
| 10 | 0.032 (0.054) | 0.017 | 0.399 |
| 20 | 0.035 (0.058) | 0.016 | 0.420 |

The test-half peak of a sequence cell is mostly a noise maximum, so dropping it removes another large deviation that doesn't replicate. Reliability rises a little more, but stays far below stim/integ cells: the median peaks at 0.035. Dropping the test peak is a selection on the test half. Removing a noise maximum from one half can only weaken that half's spurious deviation, not create agreement between the halves, so this is a reasonable upper bound.

Average z-scored rate at the train-half peak bin, relative to the row mean:

| | train | test | test / train |
|---|---|---|---|
| sequence | 6.45 | 0.16 | 2.5% |
| stim/integ | 4.95 | 1.35 | 27% |

### Takeaways

- Removing the diagonal barely changes sequence-cell reliability. The median goes from 0.016 to 0.025, flat beyond w ≈ 4, against 0.018 for random removal. Also removing the test-half peak gives 0.035 at most.
- Outside their train peak, sequence cells have essentially no reproducible structure on these six trial types. Their low reliability is not caused by the train-only peak swamping a reliable faint signal.
- The faint sequence does replicate, but weakly. In the peak-aligned average, the test half shows a small bump about ±10 bins wide at the train peak: 0.16 z, against a standard error of about 0.01 over 12k cells. That is about 2.5% of the train peak, compared with 27% for stim/integ cells. The two halves are disjoint trials and cells were selected on the train half only, so this bump is not a selection artefact.
- For the noise match, the sequence target barely moves (mean 0.044 → 0.048). The noise-matched results above stand.
- Side note: the train peak-aligned average has sharp steps at ±6 bins. Together with the noise autocorrelation reaching 0 at lag 6, this suggests the PETHs are boxcar-smoothed over about 6 bins. Confirmed in `concat_PETHs`: see "Bin size and smoothing" below.

## Rasters with peaks removed, and where the peaks fall

`plot_peak_masked_rasters` (also run by `--peak-mask`) shows sequence cells over the six trial types, with ±6 bins around both the train-half and the test-half peak removed (red). Output: `raster_seq_peaks_removed.{png,pdf}` and `raster_seq_test_sorted.{png,pdf}`.

Rendering note: removed bins are NaN. With the default downsampling, a screen row covering about 16 cells turns fully red if any one of them is NaN, so whole trial types looked removed. Colours are now resampled after mapping (`interpolation_stage='rgba'`), so red intensity shows the fraction removed.

- **Train and test, peaks removed, stack order (Rastermap fit on train):** once the train peaks are cut out, no diagonal is left in the train half. The test half shows at most a faint trace along the same path.
- **Test half sorted by a Rastermap fit on the unmasked test half:**
  - The test half gets its own sharp diagonals, as clean as the train ones.
  - They sit in the same three trial types as the train diagonals: change_b, L_sL_cR_b and R_sR_cL_b. They are built from the test-half peaks, which are almost all removed by the mask.
  - Sorting either half of these cells by peak produces a diagonal.

### Peaks land in the noisiest trial types

Over the six trial types (block_change, L_sL_cL_b, L_sL_cR_b, R_sR_cR_b, R_sR_cL_b, mistake):

| | train peak in each type | test peak in each type | mean within-type SD, test (by type) |
|---|---|---|---|
| sequence | .34 / .01 / .30 / .01 / .30 / .04 | .32 / .02 / .30 / .02 / .26 / .07 | 1.28 / 0.66 / 1.29 / 0.68 / 1.24 / 0.90 |
| stim/integ | .31 / .04 / .30 / .02 / .27 / .06 | .32 / .04 / .30 / .02 / .28 / .06 | 1.37 / 0.94 / 1.36 / 0.92 / 1.36 / 1.02 |

| | same peak trial type in both halves (chance) | peak-time r within the same type | median shift between halves |
|---|---|---|---|
| sequence | 0.35 (0.28) | 0.12 | 19 bins |
| stim/integ | 0.33 (0.27) | 0.59 | 7 bins |

- Rows are z-scored over the whole stack, and change_b, L_sL_cR_b and R_sR_cL_b have about twice the within-type SD of the other three, in both halves and in both groups. These are the block-change and incongruent trial types, and they have about 5x fewer trials (confirmed below).
- A row's maximum therefore almost always falls in one of these three types, whether the row is signal or noise. That is why the sequence diagonals sit exactly there.
- Within a type, a sequence cell's peak time barely carries over between halves (r = 0.12; a median shift of 19 bins is close to the roughly 21 expected by chance). Stim/integ peaks do carry over (r = 0.59).
- Working hypothesis: the train-half diagonals of the sequence cells are mostly the peaks of noise in the fewest-trial trial types, sorted by peak time. What replicates is a small bump (0.16 z, about 2.5% of the train peak). Selecting cells on noise maxima in one half and showing them in the other gives exactly this "present in train, gone in test" picture.

### Split-half r without the noisy trial types

`type_subset_reliability` (also run by `--peak-mask`). Output: `type_subset_reliability.json`, which also has the mean within-type SD for all 21 trial types.

| trial types | sequence median (mean) | stim/integ median (mean) |
|---|---|---|
| six analysed types | 0.016 (0.044) | 0.396 (0.408) |
| three noisy (change_b, L_sL_cR_b, R_sR_cL_b) | 0.000 (0.032) | 0.314 (0.337) |
| three quiet (L_sL_cL_b, R_sR_cR_b, mistake) | 0.063 (0.089) | 0.614 (0.555) |
| all 21 types | 0.077 (0.103) | 0.482 (0.483) |
| all except the noisy three and their movement-aligned copies (15 types) | 0.154 (0.191) | 0.662 (0.619) |

- The movement-aligned copies are noisy too: about 1.25 SD, against about 0.65 for the congruent types. The trial-type-independent periods are quieter still (inter_trial, block, quiescence, choice: 0.36-0.56). mistake and fback0 sit in between, at about 0.9, which also fits rarer trials.
- The noisy types have about 5.1-5.6x fewer trials (see below). Pure 1/√n would give an SD ratio of 2.3-2.4; the observed 1.9 is somewhat lower, because the SD also includes signal and the rows are z-scored over all types.
- Dropping the noisy types raises sequence-cell reliability about 4x on the stimulus-aligned types (0.016 → 0.063) and to 0.154 over the 15 remaining types. So sequence cells do carry reproducible structure, mostly outside the trial types where their diagonals sit.
- Their reliability is still about 4-10x below stim/integ cells on the same subsets. The noisy types pull stim/integ reliability down too (0.61 → 0.40 when they are included).
- The noise match above was done on the six types. A version restricted to the quiet types would have a higher target, about 0.06-0.09.

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
- **The train-peak carry-over is not larger than in matched cells.** (2026-11-05 correction: this section originally said "no sign of a sequence". That overstated it. The faint test-half diagonal is real signal, and the cross-half analysis below shows it tiles time. What this control shows is only that the train-peak measure doesn't single out sequence cells.) Partners that pass the same peak selection (same peak type, similar height) carry their peaks over at least as well as sequence cells: test height at the train peak 0.39 vs 0.16, peak-time r 0.20 vs 0.11. With off-peak reliability also matched, the partners still match or beat sequence cells (0.23 vs 0.16; r 0.12 vs 0.11).
  - So the faint test-half bump (0.16 z) is no larger than what any low-rate cell selected on its train peak shows.
  - It is a little smaller, because sequence cells have the least real signal of all.
- **Rastermap on the partners finds no diagonals.** The fr+region partners, given their own Rastermap fit on the train half, sort into blocky responses that reproduce in the test half (`fr_control_rasters.png`, column 2). There is no diagonal.
  - Sorting either set by train peak time gives an equally sharp diagonal that is mostly gone in the test half (columns 3-5).
  - In those sorts, the change_b rows also light up in block_change_m in both sets. That is the same trials counted in both windows, not a property of sequence cells.
- **Reading:** the sharp train-half diagonals that defined the clusters are noise peaks in the fewest-trial types. Sequence cells are also the least reliable low-rate cells overall. Their weak but reproducible timing is analysed in "Cross-half bin x bin correlation" below. Why they are less reliable than other cells at the same rate (for example bursty or unstable units, or multi-unit contamination) is not known; unit QC metrics would show it.

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
- **Conclusion (single-cell peaks only):** carry-over of each cell's train peak is no larger than in matched cells, in either alignment. Per-cell peaks are too noisy to settle the question. The population cross-half analysis below does show a preference: sequence-cell timing is stimulus-locked.

## Cross-half bin x bin correlation

Added 2026-11-05. Script: `seq_crosshalf.py`. Outputs: `dmn/figs/figure5_new/crosshalf/`, which holds `crosshalf_concat.png`, `crosshalf_profiles.png`, `crosshalf_families.png`, `crosshalf_cross_type.png` and `summary.json`. The run takes a few minutes, mostly the bootstraps.

Why: the question is not whether each cell's sharp train peak replicates. It is whether what is left in the test half is a weak but consistent tiling of time during the stimulus. The within-test-half bin x bin matrices can't show that: their diagonal band is the noise autocorrelation from the 12.5 ms boxcar (see above).

The fix is to correlate the train half at bin t with the test half at bin t', across cells: C(t, t'). The halves are disjoint trials, so their noise is independent and the band disappears. Any diagonal left in C is reproducible timing. Before correlating:

- each cell's mean is removed per segment and half, so level differences between trial types don't count
- each column is z-scored across cells, which removes the population-mean time course

From C, per trial type:

- **ridge:** mean C by |t − t'|, minus the mean at lags 15-30; its height at lag 0 and its half width
- **decoding:** for each test bin, the train bin with the highest C (a template match), counted as correct within ±2 bins; chance is 5/72 = 0.07
- **cross-type:** the same ridge between the train half of type a and the test half of type b

Confidence intervals are 2.5-97.5 percentiles over 100 bootstraps over insertions.

### Sequence cells tile time reproducibly during the stimulus

Ridge height and decoding accuracy, stimulus-aligned (95% CI in brackets):

| | change_b | L_sL_cL_b | L_sL_cR_b | R_sR_cR_b | R_sR_cL_b | mistake |
|---|---|---|---|---|---|---|
| sequence, height | 0.019 [0.011, 0.027] | 0.117 [0.099, 0.131] | 0.029 [0.024, 0.036] | 0.100 [0.089, 0.112] | 0.031 [0.023, 0.036] | 0.040 [0.030, 0.049] |
| sequence, decoding | 0.21 | 0.65 | 0.40 | 0.53 | 0.40 | 0.49 |
| fr+region, height | 0.041 | 0.193 | 0.046 | 0.174 | 0.053 | 0.062 |
| fr+peak, height | 0.057 | 0.243 | 0.063 | 0.237 | 0.079 | 0.089 |
| stim/integ, height | 0.351 | 0.768 | 0.360 | 0.750 | 0.354 | 0.537 |
| stim/integ, decoding | 0.67 | 0.94 | 0.58 | 0.93 | 0.83 | 0.82 |

- **There is a significant cross-half ridge in every trial type.** From the test half alone, the time bin can be decoded at 3-9x chance. So the faint test-half diagonal is real and tiles the 150 ms after the stimulus. This replaces the "no sequence" reading of the firing-rate control above.
- **It is temporally structured, not a single slow component.** The ridge half width is about 10-12 bins (21-25 ms). Two 6-bin boxcars alone would give about 3. The other groups have about the same half width (stim/integ about 9-10 bins).
- **It is strongest in the congruent types, not in the families' own diagonal types.** On their own type, the families' ridges are the weakest of the six:

  | family | ridge on own type | ridge in L_sL_cL_b / R_sR_cR_b |
  |---|---|---|
  | R_sR_cL_b | 0.047 | 0.12 / 0.10 |
  | L_sL_cR_b | 0.034 | 0.11 / 0.08 |
  | change_b | 0.028 | 0.12 / 0.12 |

  - The reproducible tiling therefore follows trial count (the congruent types have about 5x more trials), not the type where the train diagonal was drawn.
  - The diagonal C(t, t) is U-shaped: strongest in the first about 20 and the last about 12 bins, weakest around bins 30-55 (`crosshalf_families.png`).
- **The congruent blocks also carry a slow early-vs-late component.** There is a warm late×late corner and cold early×late corners, in sequence cells as in every other group.
- **It is stimulus-locked.** The ridge is lower in every movement-aligned copy (for example 0.069 vs 0.117 in L_sL_cL_b), and decoding falls to 0.12-0.19. The movement copies have more trials, so the drop isn't noise. The ratio is close to the partners' (0.14 vs 0.19) and to stim/integ (0.49 vs 0.77).
- **Not specific to sequence clusters.** Rate-matched partners have the same tiling at about 1.7-2x the height, with the same width and the same type pattern. Sequence cells look like the weakest end of a brain-wide stimulus-locked response of low-rate cells; nothing marks them as a separate sequence population.

### Does the tiling carry trial-type information? Side, yes

Ratio of the cross-type ridge to √(within_a · within_b). It is 1 if a cell's time course is the same in both trial types.

| | pooled (all pairs except change_b-mistake) | same side, other block | same block, other side | other side and block |
|---|---|---|---|---|
| sequence | 0.81 [0.76, 0.88] | 0.84 [0.77, 0.94] | 0.69 [0.63, 0.76] | 0.70 [0.64, 0.75] |
| fr+region | 0.82 [0.79, 0.86] | 0.91 [0.87, 0.96] | 0.76 [0.71, 0.81] | 0.71 [0.68, 0.76] |
| fr+peak | 0.80 [0.78, 0.83] | 0.91 [0.89, 0.95] | 0.72 [0.68, 0.76] | 0.67 [0.64, 0.70] |
| stim/integ | 0.95 [0.94, 0.96] | 0.96 [0.96, 0.98] | 0.93 [0.92, 0.95] | 0.91 [0.90, 0.93] |

Pairs of the four stimulus types: same side, other block = L_sL_cL_b–L_sL_cR_b and R_sR_cR_b–R_sR_cL_b; same block, other side = L_sL_cL_b–R_sR_cL_b and L_sL_cR_b–R_sR_cR_b; both differ = the congruent pair and the incongruent pair.

- **About 20% of the tiling is trial-type specific, mostly by side.** About 30% of the ridge is lost when the side differs (with or without a block change), and much less when only the block differs. These are correct trials, so stimulus side and choice side can't be separated here.
  - Rate-matched cells carry the same amount of side information.
  - Stim/integ cells carry much less: their response is nearly the same in every type (0.91-0.96), as the overall-r analysis found.
- **Block: a hint only.** The same-side, other-block ratio is 0.84 for sequence cells against 0.91 for both partner sets. The CIs overlap, so this isn't established. It fits the block-R level coupling seen in the overall-r decomposition, and it is the place to look for block information specific to sequence cells (see the open items).
- **change_b × mistake is an artefact.** That pair's ridge is 3.4x the prediction from its two within-type ridges (2.1x and 1.8x in the partners). That is the shared trials: a trial in the train list of one type can be in the test list of the other, so the same spikes sit on both sides. It is excluded from the pooled ratio.

## Spike-thinned stim/integ cells

Added 2026-11-13. Scripts are in `stint_thin/`. Status: written and tested locally, not yet run on ORCD.

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

Next, after the run: z-score the thinned rows and rerun the `seq_noise_match.py` metrics with thinned stim/integ cells next to the Gaussian-noise ones. Also fit Rastermap on the thinned train half, to see whether train-only diagonals appear.

## Open items / next steps

- Block information in sequence cells, tested directly: blockL vs blockR segments (cross-half, as above), and the per-cell level coupling between trial types that share a block.
- Separate stimulus side from choice side in the side information. This needs error trials split by side, which the stack doesn't have (mistake_s pools both sides).

- Unit quality of sequence cells vs fr-matched cells: presence ratio, amplitude, drift, ISI violations. This would explain why sequence cells are less reliable than other cells at the same rate. It needs the clusters tables, not `brainwidemap`.
- Null for the sequence picks: take a stack of pure noise matched in per-type SD and smoothing, run the same Rastermap fit, and see whether diagonals of similar sharpness and cell count appear. Partly answered by the firing-rate matched control: fr-matched cells sorted by train peak give equally sharp diagonals. What is still open is whether the whole-stack Rastermap would cluster pure noise rows into diagonal clusters.
- ~~Find the code that built the odd/even stack.~~ Done 2026-11-13: upstream `dmn_bwm._oddeven_means_concat`, by position within each type's list (see "Spike-thinned stim/integ cells").
- Unit quality: `bwm_tables/clusters.pqt` has presence ratio, drift, contamination, amplitude and sliding-RP violations per uuid, so this needs no job.
- Significance of the spread-over-time difference: bootstrap over insertions (`pid`), then cells, with fresh noise each iteration, plus a label permutation within insertions. Use the thirds over bins 6-65 to avoid edge effects. Deferred for now.

- Check whether the sequence same-type block structure is per-cell trial-type offsets. Compare the variance of segment means between groups, and redo the across-type analysis after subtracting each cell's per-segment mean.
- Run `--data old` for the cached stack.
- The noise is modelled at the half-average level. If single-trial PETHs can be rebuilt (this needs `brainwidemap` and the raw insertions), add noise per trial and re-average, and check whether trial counts per trial type differ between the groups.
- Opposite control: average or denoise sequence cells, for example by fitting on train and testing on the projected test half, to see whether any shared cross-type structure appears.
- Remove the noise-induced diagonal band from the cross-correlograms before comparing them, for example by subtracting the cross-correlogram of the train-minus-test noise.
