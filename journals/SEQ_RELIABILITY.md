# Reliability of sequence cells

Started 2026-11-19. The split-half and cross-trial-type reliability sections were moved here from [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md), unchanged apart from links. Cell selection: [FIG5_CLUSTERS.md](FIG5_CLUSTERS.md). Noise-matched, spike-thinned and firing-rate controls, trial membership per type, and bin size: [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md).

## Question

How reproducible are the sequence cells' responses: between disjoint halves of the trials (odd vs even), across trial types, and, next, across stimulus contrast? And among the most reliable sequence cells, is the correlation structure (which cells are active when, and which cells co-vary) kept across trial types and contrasts, compared with other cells?

## Terms used throughout

- **Halves:** train = odd trials (1st, 3rd, …), test = even trials, by position within each trial type's own trial list (`dmn_bwm._oddeven_means_concat`).
- **Six analysed types:** the stimulus-aligned types of `fig5.CONDITIONS`, [0, 150] ms after stimulus onset, 72 bins each (2 ms stride, 12.5 ms boxcar): change_b, L_sL_cL_b, L_sL_cR_b, R_sR_cR_b, R_sR_cL_b, mistake. Subscripts: s = stimulus side, c = choice, b = block.
- **Split-half r:** per cell, Pearson r between the train and test PETH over the bins named (the 432 bins of the six types unless stated).
- **Cross-half C(t, t′):** correlation across cells of train at bin t with test at bin t′ (see "Cross-half bin x bin correlation").

## Reliability with the train-half diagonal removed

Run: `python seq_noise_match.py --peak-mask` (`run_peak_mask`). Outputs: `peak_masked_reliability.{png,json}` in `figs/figure5_new/noise_match/`.

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
- For the noise match, the sequence target barely moves (mean 0.044 → 0.048). The noise-matched results in [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md) stand.
- Side note: the train peak-aligned average has sharp steps at ±6 bins. Together with the noise autocorrelation reaching 0 at lag 6, this suggests the PETHs are boxcar-smoothed over about 6 bins. Confirmed in `concat_PETHs`: see "Bin size and smoothing" in [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md).

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

- Rows are z-scored over the whole stack, and change_b, L_sL_cR_b and R_sR_cL_b have about twice the within-type SD of the other three, in both halves and in both groups. These are the block-change and incongruent trial types, and they have about 5x fewer trials (confirmed in [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md), "Trial membership and counts").
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
- The noisy types have about 5.1-5.6x fewer trials ([SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md), "Trial membership and counts"). Pure 1/√n would give an SD ratio of 2.3-2.4; the observed 1.9 is somewhat lower, because the SD also includes signal and the rows are z-scored over all types.
- Dropping the noisy types raises sequence-cell reliability about 4x on the stimulus-aligned types (0.016 → 0.063) and to 0.154 over the 15 remaining types. So sequence cells do carry reproducible structure, mostly outside the trial types where their diagonals sit.
- Their reliability is still about 4-10x below stim/integ cells on the same subsets. The noisy types pull stim/integ reliability down too (0.61 → 0.40 when they are included).
- The noise match in [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md) was done on the six types. A version restricted to the quiet types would have a higher target, about 0.06-0.09.

## Per-cell consistency across trial types

Added to this journal 2026-11-19 from outputs written 2026-11-09 (`seq_rasters.py --consistency`; `figs/figure5_new/seq_rasters/consistency.json`, `r_test.json`); these numbers were not journaled at the time. Definitions are from the script:

- **r_within:** mean Pearson r over the 15 pairs of the six types between a cell's 72-bin segments of the trial-averaged PETH (mean of the two halves).
- **r_cross:** the same, but the train segment of type a against the test segment of type b (a ≠ b). The two never share trials, so noise cannot inflate r. change_b × mistake is left out because those types share trials.
- **r_test:** the same within the test half only (change_b × mistake left out). Used to rank cells in `seq_rasters.py` and `seq_top_compare.py`.

| | cells | r_cross median (mean) | r_cross 90th pct | r_within median | r_test median | r_test > 0.2 | r_test > 0.5 |
|---|---|---|---|---|---|---|---|
| sequence | 12,084 | 0.022 (0.041) | 0.154 | 0.037 | 0.019 | 8.6% | 0.9% |
| stim/integ | 3,620 | 0.411 (0.425) | 0.836 | 0.567 | 0.403 | 72% | 40% |
| other | 37,488 | 0.062 (0.131) | 0.424 | 0.104 | 0.067 | 26% | 7.7% |

The top 3,615 sequence cells by r_test (as many as stim/integ cells) have median r_test 0.139. `seq_rasters.py` PICKS are sequence cells above the median sequence firing rate with r_cross 0.55-0.98 (top 2%); PICKS_TYPICAL sit at the median.

Not yet journaled, outputs on disk only (no claims made from them here): `seq_top_compare.py` (`seq_rasters/top_compare/`: the fig5 metrics for the top-r_test sequence and other cells), `seq_pair_xcorr.py` (`pair_xcorr/`: pair cross-correlograms per type and their consistency across types), `seq_timing.py` (`timing/`: peak-latency distributions and their correlation across types).

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

- **There is a significant cross-half ridge in every trial type.** From the test half alone, the time bin can be decoded at 3-9x chance. So the faint test-half diagonal is real and tiles the 150 ms after the stimulus. This replaces the "no sequence" reading of the firing-rate control in [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md).
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
## Reliability by stimulus contrast

Added 2026-11-19. Status: job written and tested locally on 6 insertions; not yet run on ORCD.

### Why a new run

The stack pools all contrasts in each trial type. Splitting every contrast level would leave too few trials (incongruent types have about 1-8 trials per session per contrast), so trials are split into two groups. Counts from `bwm_tables/trials.pqt` (345 sessions of the stack, same masks and type definitions as the stack):

| median trials per session, both halves | change_b | L_sL_cL_b | L_sL_cR_b | R_sR_cR_b | R_sR_cL_b | mistake |
|---|---|---|---|---|---|---|
| contrast 0 | 2 | 9 | 1 | 10 | 1 | 19 |
| 6.25% | 5 | 23 | 4 | 23 | 4 | 24 |
| 12.5% | 4 | 28 | 6 | 26 | 5 | 11 |
| 25% | 6 | 32 | 7 | 26 | 6 | 4 |
| 100% | 5 | 34 | 7 | 31 | 8 | 1 |

Groups:

- **low:** 0, 6.25 and 12.5% (contrast 0 means no visible stimulus; its "side" is the side the task assigned)
- **high:** 25 and 100%
- **low_nz:** low without the zero-contrast trials, to check that the no-stimulus trials don't drive a low/high difference
- **all:** every contrast; must reproduce the stack's six segments exactly

Low and high have about equal trial counts in the five stimulus and block-change types (about 11-14 per session in change_b and the incongruent types, about 60 in the congruent types). Mistake is the exception: about 54 low against about 5 high per session, so mistake/high has about 2-3 trials per half and should be left out of most comparisons.

### The job

Code in `contrast_split/`, following `stint_thin/` (same insertion cache, `trials.pqt`, binning and odd/even assignment; no ONE or Alyx).

- No cell selection. The job takes every good unit (`label == 1` in `bwm_tables/clusters.pqt`) of the 515 stack insertions (`stint_thin/data/insertions.csv`), about 55,000 units, and keeps their uuids. The sequence and stim/integ labels are attached locally by joining on uuid with the stack (`concat_cvTrue.npy`), and the same join checks that group `all` reproduces the stack.
- `contrast_peths.py compute` handles one shard of insertions (every n-th insertion of the sorted list; the 4 default shards hold 129/129/129/128 and together cover all 515 once). Each type's trials are binned once. Every contrast group is then a subset of those binned trials, and its odd/even halves are taken by position within the group's own trial list, as the stack does within each type. Output `shards/{pid}.npz`: P0/P1 (units × 4 groups × 6 types × 72 bins = 1,728 columns), trial counts per group, type and half, uuid and `cluster_id`. A half with no trials is NaN.
- `contrast_peths.py stack` concatenates the shards into one file, `OUT/contrast_peths.npz`: P0, P1 (units × 1,728, raw rate, float32; columns labelled by `col_group` and `col_type`), `ntrials` (units × 4 × 6 × 2), uuid, pid, `cluster_id`. About 760 MB uncompressed, saved compressed.
- **Is `all` needed?** Not for the analysis; it is the check against the stack, which only stores the train half z-scored. Since the groups are cut from the same binned trials, it costs no extra binning, only a quarter of the output.
- Local test (alyx-host insertion cache, the 6 cached stack insertions, 700 good units, 696 of them in the stack). Time per insertion: 0.0-0.5 s to load the spikes and 0.2-0.3 s to bin all four groups, so about 5 min for all 515 in one job; peak memory 1.5 GB. Results: group `all` equals `concat` exactly on the six segments (max |diff| 0.0), and its odd half correlates with `concat_z_train` at r = 1.0000 for every unit with spikes (18 flat rows). low + high trial counts add up to `all` in every type. The 4 good units not in the stack are kept; the join drops them.
- Run on ORCD (the agent does not submit). Commit and pull there first, then:

  ```bash
  bash contrast_split/submit_contrast.sh
  ```

  Defaults: 4 shards on `pi_fiete` at 4 G with a 1 h limit, then the stack job (8 G). Expect a few minutes per shard; reading the spike caches from ORCD storage may dominate. Output goes to `$ONE_CACHE_DIR/dmn/res/contrast_split/`. `PARTITION=mit_preemptable` adds `--requeue`; shards skip insertions already written. Then rsync `OUT/contrast_peths.npz` (not `shards/`) to `~/Downloads/contrast_split/`.

### Plan for the analysis

One step at a time; each step reports back before the next.

1. **Check the run.** Join on uuid with the stack: `all` must equal `concat` (even half) on the six segments and correlate at r = 1 with `concat_z_train` (odd half). Per group: trial counts per type (median, fraction of cells with < 3 trials in a half), fraction of rows with no spikes in a half, and whether `all` reliability reproduces the stack numbers above (sequence median 0.016, stim/integ 0.396).
2. **Split-half reliability by contrast.** Per cell and group, split-half r on the six types and per type, for sequence, stim/integ, other and the fr+region partners from `seq_fr_control.py`. Rows z-scored per group over its 432 bins (r itself does not depend on it).
   - Fairness: low and high differ in response strength, not only in trial count. Compare per type where counts match (all but mistake), and report the per-type noise variance (train minus test) next to r so a reliability change can be split into signal change vs noise change.
   - Figure: per-cell reliability, low vs high scatter per group, with the medians.
3. **Cross-half C(t, t′) per group and across groups.** Rerun the `seq_crosshalf.py` ridge and decoding per contrast group, plus C between the train half of one group and the test half of the other (independent trials). Ratio of the cross-contrast ridge to √(ridge_low · ridge_high): 1 means the same time course at both contrasts. Also look for an off-diagonal shift: lower contrast should delay visual responses, so report the lag of the cross-contrast ridge peak.
4. **Most reliable cells, selected without circularity.** Rank cells by reliability in one contrast group and evaluate them in the other (and the reverse); never select and evaluate on the same trials. Sets: top-n sequence, top-n other at the same n, and the same n of other cells matched in firing rate and region (the fr+region partners), and stim/integ.
5. **Correlation structure across trial types and contrast.** For each set, two kinds of structure, both computed across halves so no shared trial noise enters:
   - **time-course similarity:** the 12 × 12 matrix (6 types × 2 contrasts) of cross-half r between whole segments (cells × 72 bins), normalised by √(rel_a · rel_b) as in "Does the tiling carry trial-type information?".
   - **cell-pair signal correlations:** within each type × contrast segment, the cell × cell matrix corr(train_i, test_j) symmetrised. Then the r between these matrices (upper triangles) for every pair of segments: are the same cells co-active in every type and at both contrasts?
   - Ceilings and nulls: the same-segment value (train vs test of one segment) is the ceiling. Null: match sets on firing rate (fr+region partners) and check one null draw against the others (several partner draws). Pairs from the same session share trial noise only within a half, so cross-half avoids it; report same- and different-session pairs separately anyway.
   - Figures: the 12 × 12 matrices per set; the matrix-similarity values per set with the partner-draw spread.

Open caveats: correct trials only in the four stimulus types, so low-contrast trials are partly guesses; mistake/high is nearly empty; contrast and trial count are only matched approximately per type.
