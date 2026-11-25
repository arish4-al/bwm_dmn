# Sequence-cell detection as in Taxidis 2020 / Dorian 2025

Started 2026-11-24. Background: [SEQ_RELIABILITY.md](SEQ_RELIABILITY.md), [SEQ_STINT_NOISE.md](SEQ_STINT_NOISE.md), [FIG5_CLUSTERS.md](FIG5_CLUSTERS.md).

## Question

Our sequence cells were picked by eye from Rastermap clusters with train-half diagonals. Which units pass the detection procedure of Dorian et al. 2025 (Cell Reports, "Hippocampal sequences represent working memory and implicit timing"; method from Taxidis et al. 2020, Neuron), and how do they relate to the Rastermap sequence and stim/integ clusters?

## Their procedure

Per cell and trial type, on the trial-averaged trace over the odor + delay window:

- pre-filter: the cell spikes in the window on at least max(10% of trials, 3) trials (Taxidis 2020)
- preferred type: the one with the larger peak of the trial average
- peak test: the maximum of the trial average must exceed the 95th percentile of 2,000 (Taxidis: 1,000) shuffled maxima; in each shuffle every trial is circularly shifted on its own by up to ±½ window, the trials are averaged and the maximum over the window taken
- trial reliability: above 2 SD within ±0.5 s of the peak on at least 20% of preferred-type trials (window 3.5 s or 6 s)
- detection uses all trials; odd/even trials are used only to show the sequence (their Fig. S1)

## Adaptation (step 1: the job)

Code: `seq_detect/seq_detect.py` (ORCD: `seq_detect/submit_seq_detect.sh`), same inputs as `contrast_split/` (insertion cache, `trials.pqt`, `clusters.pqt`; no ONE or Alyx). Every good unit of the 515 stack insertions (55,021), the six stimulus-aligned types, [0, 150) ms.

Per unit, type and trial set (all, odd, even, by position in the type's own list as the stack):

- **statistic:** max over the 72 columns of the trial-summed PETH (12.5 ms boxcar at the stack's column starts).
- **null:** 2,000 draws; every trial's spikes shifted circularly on its own within [0, 150) ms by a random offset, uniform over the 300 offsets of the 0.5 ms grid (equivalent to ±75 ms), then the same statistic. Columns are circular for the data too (the last ~6 columns wrap instead of running to 160 ms), so data and null differ only in the shifts. The all-trials draw is the sum of the odd and even draws.
- **p** = (1 + #{null ≥ data}) / 2,001. **p_corr:** the max over the six types handled by comparing the minimum p over types with the same minimum per draw.
- **p0, p0_corr:** calibration, draw 0 tested as if it were data against the other draws.
- **n_active:** trials with ≥ 1 spike in [0, 150) ms (for the pre-filter).
- **rel:** fraction of trials with ≥ 1 spike within ±10, ±20 or ±30 ms of the peak column's centre, for (evaluated set @ peak set) = all@all, odd@odd, even@even, even@odd, odd@even. ±20 ms is the ±0.5 s-of-3.5 s window scaled to 150 ms; "≥ 1 spike" replaces "above 2 SD".
- also kept: odd/even PETHs, circular (`P_circ`) and exactly as the stack (`P_exact`).

### Local test (6 cached insertions, 700 units, 696 in the stack)

| check | result |
|---|---|
| `P_exact` vs the stack's binning code (`bin_peth`), all 6 insertions × 6 types | max \|diff\| 0 |
| even half vs `concat` (six types, joined on uuid) | max \|diff\| 0 |
| odd half vs `concat_z_train` | r ≥ 0.9999999 (678 units with spikes) |
| stored max = n_trials × max(`P_circ`) | to float32 precision |
| calibration: fraction of units with p0 < .05, per type (all trials) | 0.010–0.030 |
| calibration: p0_corr < .05 (all / odd / even) | 0.031 / 0.039 / 0.030 |
| time, memory | about 0.06 s per unit; 1.4 GB peak (234 units) |

The calibration is below 5%: spike counts are discrete, so maxima tie, and p counts ties against the data. The test is somewhat conservative, more so at low rates. Pass rates on these 700 units are not interpreted (6 insertions).

### Run on ORCD (the agent does not submit)

Commit and pull there first, then:

```bash
bash seq_detect/submit_seq_detect.sh
```

16 shards (2,850–4,290 units each, about 5 min) at 8 G and 2 h, then the stack job. Output: `$ONE_CACHE_DIR/dmn/res/seq_detect/seq_detect.npz` (about 50 MB); rsync it to `~/Downloads/seq_detect/`.

## Step 2: detection

Added 2026-11-24. Script: `seq_detect_step2.py`. Output: [figs/seq_detect/step2.png](figs/seq_detect/step2.png), `step2.json`. Input: `~/Downloads/seq_detect/seq_detect.npz` (all 55,021 units, run on ORCD 2026-11-24, commit b9e5910).

Criteria (window ±20 ms, reliability threshold 20%, α = .05):

- **paper:** preferred type = the type with the highest peak of the all-trial average; on it, pre-filter (spikes on ≥ max(10%, 3) trials), p < .05, and ≥ 1 spike within ±20 ms of the peak on ≥ 20% of trials. All on the same trials, as published.
- **type-corrected:** preferred type = the type with the smallest p (ties: higher peak), tested with p_corr (minimum p over the six types against its shuffle distribution); pre-filter and reliability on that type. Added 2026-11-24, replacing a first version that used p_corr with the highest-peak type (its numbers are dropped).
- **cross-validated:** preferred type, pre-filter, p and reliability on odd trials; then on even trials p < .05 on that type and ≥ 20% of trials active within ±20 ms of the odd-trial peak. Done for both the paper and the type-corrected criterion.
- **null:** the same criterion with p0 (shuffle draw 0 tested as data) in place of p. It calibrates the peak test only; the pre-filter and reliability are those of the real data.

Groups: fig5 classes by uuid; fr+region partners = one partner per sequence cell, nearest log rate in the same Beryl region (from the contrast step; 5 draws, 98% the same cells, so the range is not a resampling spread).

| group | units | median rate (Hz) | paper | its null | cross-validated (units) | type-corrected | its null | type-corrected, cross-validated (units) |
|---|---|---|---|---|---|---|---|---|
| sequence | 12,182 | 2.4 | 6.0% | 1.0% | 0.2% (24) | 6.8% | 1.6% | 1.2% (149) |
| fr+region partners | 12,182 | 2.3 | 9.0% | 0.9% | 1.6% (192) | 12.8% | 1.5% | 4.8% (582) |
| stim/integ | 3,632 | 12.3 | 54.4% | 2.8% | 20.4% (741) | 72.5% | 4.3% | 47.7% (1,734) |
| other | 38,755 | 5.9 | 18.5% | 1.8% | 4.3% (1,667) | 27.0% | 2.6% | 12.8% (4,973) |

Rate = mean over the six types and both halves. Percent = fraction of the group passing.

![step 2](figs/seq_detect/step2.png)

Panels: (1) pass rates, solid = all trials, faint = cross-validated, hatched = type-corrected, black = null; (2) paper pass rate against firing rate, bins = sextiles of the sequence cells' rate; (3) preferred type of the paper-passing units; (4) their peak times.

- **The published criteria pass fewer Rastermap sequence cells than rate- and region-matched cells** (6.0% vs 9.0%), and far fewer than stim/integ (54%) or other cells (19%). Above about 2 Hz the sequence cells pass less often than partners at the same rate (panel 2).
- **Cross-validated, almost none remain:** 24 sequence cells (0.2%), against 1.6% of partners and 20% of stim/integ. Most sequence cells that pass on all trials do not peak reproducibly on held-out trials.
- **The preferred type is mostly one of the three low-trial types** (block change, the two incongruent types) in every group, most of all for sequence cells (87% of passing cells, against 68% for partners and other). The max-of-average rule favours noisy types, as the Rastermap selection did.
- **Type-corrected, cross-validated: 149 sequence cells (1.2%), against 4.8% of partners.** Choosing the type by p instead of by peak height moves the preferred type to the congruent types with many trials (LLL + RRR: 64% of type-corrected passing sequence cells, against 6% under the paper rule), and about 6x more sequence cells survive held-out testing. The gap to partners narrows from about 1:8 to 1:4 but remains.
- The null passes 1–3% of units, below the 5% nominal, so the peak test is conservative (tied spike counts).

Open caveats: the null covers the peak test only; the ±20 ms window and "≥ 1 spike" are scalings of calcium criteria; peaks in partners and other cells pile up in the last ~20 ms (panel 4), not checked (the columns are circular there).

## Step 3: the sequence on held-out trials

Added 2026-11-24. Script: `seq_detect_step3.py`. Output: [figs/seq_detect/step3.png](figs/seq_detect/step3.png), `step3.json`.

- **Units:** those passing the paper's criteria on odd trials only (preferred type, pre-filter, p < .05, reliability; "odd" in step 2). Partners: draw 0, each unit counted once.
- **Rows:** the unit's preferred-type PETH (chosen on odd trials), scaled 0–1 per row and half, sorted by the odd-trial peak. Odd trials (the sort, a diagonal by construction) next to even trials in the same order, as in Dorian 2025 Fig. S1.
- **Peak reproducibility:** odd vs even peak on the preferred type. r = Pearson r over units; near = |Δpeak| ≤ 5 columns (about 10 ms). Null: each unit's odd peak paired with another passing unit's even peak (2,000 permutations), which keeps both peak distributions. p ≤ .001 is the permutation floor.
- **Early only:** units whose odd peak is before the last ~20 ms (column < 62, centre < 136 ms), since many peaks sit at the window end.

| group | units | r, odd vs even peak | near (null) | p | units, early peaks | r, early | near, early (null) |
|---|---|---|---|---|---|---|---|
| sequence | 652 | 0.23 | 21% (15%) | ≤.001 | 537 | 0.22 | 22% (16%) |
| fr+region partners | 748 | 0.56 | 42% (19%) | ≤.001 | 499 | 0.59 | 39% (16%) |
| stim/integ | 1,432 | 0.82 | 64% (18%) | ≤.001 | 1,318 | 0.81 | 66% (19%) |
| other | 4,847 | 0.65 | 48% (20%) | ≤.001 | 3,183 | 0.65 | 44% (17%) |

![step 3](figs/seq_detect/step3.png)

- **On held-out trials the sequence cells' diagonal is gone** (top left, right half): their even-trial peaks barely follow the odd-trial order (r = 0.23; 21% within 10 ms, against 15% by chance).
- **Rate-matched partners passing the same criteria keep much more of their diagonal** (r = 0.56, 42% within 10 ms), stim/integ cells keep nearly all of it (r = 0.82), and other cells sit in between (0.65). This holds without the end-of-window peaks (early columns).
- So detecting with the published procedure and validating on odd/even trials, as Dorian 2025 do, gives a reproducible sequence for stim/integ and other cells, and a weak one for the Rastermap sequence cells, weaker than for cells of the same rate and region.

Open caveats: |Δpeak| is not circular, though the columns are (odd peaks near 140 ms with even peaks near 10 ms count as far apart; visible in the sequence scatter); one partner draw; 33% of partner and other peaks are in the last ~20 ms (late ramps, not examined).

### Type-corrected units

Added 2026-11-24. `python seq_detect_step3.py odd_tc`. Output: [figs/seq_detect/step3_tc.png](figs/seq_detect/step3_tc.png), `step3_tc.json`. Same figure and measures as above; units pass the type-corrected criterion on odd trials, and each row is the type with the smallest odd-trial p.

| group | units | r, odd vs even peak | near (null) | p | units, early peaks | r, early | near, early (null) |
|---|---|---|---|---|---|---|---|
| sequence | 682 | 0.48 | 37% (16%) | ≤.001 | 531 | 0.42 | 36% (16%) |
| fr+region partners | 1,161 | 0.74 | 55% (23%) | ≤.001 | 739 | 0.75 | 52% (17%) |
| stim/integ | 2,263 | 0.89 | 70% (17%) | ≤.001 | 2,046 | 0.89 | 72% (17%) |
| other | 8,035 | 0.79 | 59% (23%) | ≤.001 | 5,165 | 0.78 | 55% (17%) |

![step 3, type-corrected](figs/seq_detect/step3_tc.png)

- **With the type chosen by p, the sequence cells keep a faint diagonal on held-out trials** (r = 0.48, against 0.23 with the paper rule; 37% within 10 ms against 16% by chance).
- **It stays weaker than in every other group** (partners 0.74, other 0.79, stim/integ 0.89), with or without the end-of-window peaks.

### Type-corrected, cross-validated units only

Added 2026-11-24. `python seq_detect_step3.py odd_tc cv`. Output: [figs/seq_detect/step3_tc_cv.png](figs/seq_detect/step3_tc_cv.png), `step3_tc_cv.json`. Units that also pass on even trials (p < .05 on the odd-chosen type, ≥ 20% of even trials active within ±20 ms of the odd peak), so the even half is partly selected and the peak agreement below is not a held-out test; it shows what the passing set looks like.

| group | units | r, odd vs even peak | near (null) | units, early peaks | r, early |
|---|---|---|---|---|---|
| sequence | 149 | 0.88 | 73% (21%) | 111 | 0.88 |
| fr+region partners | 579 | 0.92 | 73% (31%) | 336 | 0.91 |
| stim/integ | 1,734 | 0.94 | 76% (17%) | 1,572 | 0.93 |
| other | 4,973 | 0.92 | 72% (28%) | 3,015 | 0.91 |

![step 3, type-corrected, cross-validated](figs/seq_detect/step3_tc_cv.png)

- The 149 sequence cells that pass on both halves form a sequence tiling 0–150 ms on both halves, with peak agreement close to the other groups (r = 0.88 vs 0.92–0.94). About a quarter peak in the last ~20 ms (partners 42%, other 39%, stim/integ 9%).

## Pooled trial types

Added 2026-11-24. Job: `POOLED=1 bash seq_detect/submit_seq_detect.sh` (commit eb35992), output `~/Downloads/seq_detect_pooled/seq_detect.npz`. Analysis: `seq_detect_step2.py pooled`, `seq_detect_step3.py odd pooled`, `seq_detect_step3.py odd cv pooled`. Figures: [step2_pooled.png](figs/seq_detect/step2_pooled.png), [step3_pooled.png](figs/seq_detect/step3_pooled.png), [step3_cv_pooled.png](figs/seq_detect/step3_cv_pooled.png).

- **Trials:** one set per unit, the union of the six stimulus-aligned types (each trial once; median 376 trials per session, against 385 summed over the six types), odd/even by position in that list. Everything else as in steps 1–3.
- With one type there is no preferred type, so the paper and type-corrected criteria are the same.
- Calibration: p0 < .05 in 3.6% (all), 3.2% (odd), 3.2% (even) of units.

| group | units | pass, all trials | null | cross-validated (units) | units passing on odd | held-out peak r | within 10 ms (null) | r, early peaks |
|---|---|---|---|---|---|---|---|---|
| sequence | 12,182 | 7.9% | 1.0% | 2.7% (333) | 695 | 0.73 | 52% (18%) | 0.69 |
| fr+region partners | 12,182 | 13.4% | 1.2% | 7.4% (900) | 1,302 | 0.82 | 63% (24%) | 0.80 |
| stim/integ | 3,632 | 78.8% | 3.4% | 66.0% (2,398) | 2,601 | 0.92 | 75% (16%) | 0.92 |
| other | 38,755 | 30.2% | 2.3% | 19.1% (7,412) | 9,527 | 0.85 | 66% (23%) | 0.82 |

Held-out peak r and within 10 ms: over units passing on odd trials, odd vs even peak (step 3); permutation p ≤ .001 in every group. Early peaks: odd peak before the last ~20 ms.

![step 2, pooled](figs/seq_detect/step2_pooled.png)

![step 3, pooled](figs/seq_detect/step3_pooled.png)

- **Pooling raises cross-validated pass rates in every group, and narrows the gap to partners.** From the six-type type-corrected version: sequence 1.2% → 2.7%, partners 4.8% → 7.4% (stim/integ 48% → 66%, other 13% → 19%). The sequence : partner ratio goes from about 1 : 4 to 1 : 2.7.
- **On held-out trials the pooled sequence cells keep a clear diagonal** (r = 0.73, 52% of units within 10 ms against 18% by chance), the strongest of any version so far, but still below partners (0.82), other cells (0.85) and stim/integ (0.92), with or without end-of-window peaks.
- **Below about 2 Hz almost no unit passes in any group** (panel 2): with ~190 trials per half, the 20% reliability criterion needs a spike within ±20 ms of the peak on ~38 trials, which low-rate units rarely reach. Above it, sequence cells pass less often than partners at the same rate.
- The cross-validated subset alone (`step3_cv_pooled.png`; even half partly selected, so not a held-out test): 333 sequence cells, r = 0.95, against 0.90–0.94 in the other groups.

Open caveats: as in step 3 (non-circular |Δpeak|, one partner draw, late-ramp peaks not examined); the rate axis of panel 2 is the pooled-trial mean rate.

### Contrast groups (job only, not yet run)

Added 2026-11-24. `seq_detect.py --contrast {low, high, low_nz}` keeps only trials whose |contrast| is in the group (groups as in `contrast_split/`: low = 0, 6.25, 12.5%; high = 25, 100%; low_nz = low without 0), applied to the trial list before odd/even, so the halves are by position in the group's own list. Shuffle seeds also depend on the group. Shards and the stacked file carry a `contrast` key; `stack` refuses mixed shards.

```bash
POOLED=1 CONTRAST=low bash seq_detect/submit_seq_detect.sh
POOLED=1 CONTRAST=high bash seq_detect/submit_seq_detect.sh
```

Output `$ONE_CACHE_DIR/dmn/res/seq_detect_pooled_{low,high}/seq_detect.npz`; rsync to `~/Downloads/seq_detect_pooled_{low,high}/`.

Local test (the 6 cached insertions, 700 units):

| check | result |
|---|---|
| no `--contrast` vs the earlier pooled test output | every array identical |
| trials, low + high = all | in all 6 insertions; low 94–171, high 101–180 trials per insertion (about 47–90 per half) |
| `P_exact` vs `bin_peth` on the low trials (`--check`, 1 insertion) | max \|diff\| 0 |
| calibration, p0 < .05 (all / odd / even) | low .026 / .023 / .020; high .023 / .013 / .020; low_nz .027 / .036 / .024 |

With about half the trials per group, the 20% reliability criterion needs a spike near the peak on about 10–18 trials per half, and the peak test has less power than in the all-contrast pooled run.

## fig5 v2 for the passing sequence cells

Added 2026-11-24; revised the same day (all groups restricted to passing units, thinned + noise removed, peak-order panel added; the first version's numbers are dropped). Script: `fig5_v2_detect.py --set {tc_cv, pooled_cv}` (uses `fig5_v2.make_figure` with new optional arguments; the default fig5 v2 redraws pixel-identical). Output: `dmn/figs/figure5_v2_detect/{set}/` and copies [figs/seq_detect/fig5v2_tc_cv.png](figs/seq_detect/fig5v2_tc_cv.png), [figs/seq_detect/fig5v2_pooled_cv.png](figs/seq_detect/fig5v2_pooled_cv.png). The step 2–3 figures were redrawn in the fig5 v2 colours (sequence #01789a, rate+region #6a3d9a, stim/integ #e58324, other #888a8c).

Every group passes the same cross-validated criterion (type-corrected six-type run, or pooled run):

- **sequence:** passing Rastermap sequence cells.
- **rate+region matched:** one partner per passing sequence cell among the passing cells that are neither sequence nor stim/integ, same Beryl region, nearest log firing rate (`seq_fr_control.match_cells`, seed 0). Type-corrected: 99% same region, no reuse; pooled: 100% same region, 1 reused partner.
- **stim/integ:** passing stim/integ cells.
- **Order:** panels a and b show rows in the global Rastermap order (fit on the train half of the whole stack). The new panel right of b sorts each group by the peak time of its train-half trial-type average and shows the test half (6 types + average).
- **Panel h test:** spread = late − early third of the all-pairs cross-half m(t); cell label shuffle (2,000, seed 8080), each control against the sequence set; p two-sided around the null mean. Calibration: two random draws of the same number of Rastermap sequence cells (any) against each other.

| | sequence | rate+region matched | stim/integ |
|---|---|---|---|
| type-corrected: units | 149 | 149 | 1,734 |
| median split-half r (6 types) | 0.30 | 0.48 | 0.64 |
| median rate (counts / 12.5 ms) | 0.105 | 0.109 | 0.183 |
| spread, all pairs | 0.039 | 0.102 | 0.278 |
| D vs sequence, p | | +0.064, .13 | +0.239, ≤.001 |
| pooled: units | 333 | 332 | 2,398 |
| median split-half r | 0.29 | 0.39 | 0.55 |
| median rate | 0.122 | 0.122 | 0.169 |
| spread, all pairs | 0.018 | 0.085 | 0.224 |
| D vs sequence, p | | +0.067, .004 | +0.206, ≤.001 |

Calibration (random sequence cells, draw 1 vs 2): p = .10 (149 cells), .63 (333 cells).

![fig5 v2, type-corrected cross-validated](figs/seq_detect/fig5v2_tc_cv.png)

![fig5 v2, pooled cross-validated](figs/seq_detect/fig5v2_pooled_cv.png)

- **Among cells that all pass the criterion, the sequence cells are the least reliable** (split-half r 0.29–0.30 vs 0.39–0.48 for rate- and region-matched passing cells), at the same rate.
- **In peak-time order all three groups show a sequence on the test half** (panel right of b); the sequence and partner sets show it mainly in the congruent types and the average, stim/integ in every type.
- **Spread is lower in the sequence set than in passing partners:** significant in the pooled set (D = 0.067, p = .004), same direction but not significant with 149 cells (D = 0.064, p = .13). Both are far below passing stim/integ (p ≤ .001).
- **Population mean (d):** passing partners rise late in every type more than the passing sequence cells; passing stim/integ have the early transient and a sustained response.

Open caveats: the sets are selected on both halves, so a–h are not held-out; partners are matched on rate and region, not reliability (sequence cells are less reliable); one partner draw; 149 cells give a noisy spread (calibration draw difference −0.019).

## All units together, and the Rastermap clusters of passing units

Added 2026-11-24. Script: `seq_detect_allcells.py`. Output: [figs/seq_detect/allcells_step3.png](figs/seq_detect/allcells_step3.png), [figs/seq_detect/allcells_clusters.png](figs/seq_detect/allcells_clusters.png), `allcells.json`.

Steps 2–3 on all 54,569 units in the stack (no group split), for the three versions, each cross-validated:

- **paper:** six-type run, preferred type = highest peak, p of that type.
- **type-corrected:** six-type run, preferred type = smallest p, tested with p_corr.
- **pooled:** pooled run, one trial set.
- **passes on odd:** the criterion on odd trials; **cross-validated:** also p < .05 and ≥ 20% of trials active near the odd peak on even trials.
- **Rastermap cluster:** `rm_labels` of the stack (Rastermap fit on the train half, 100 clusters numbered along the sort), joined on uuid; class = fig5 sequence clusters, stim/integ clusters 22–28, or other.

| version | passes on odd (null) | cross-validated | of those: sequence clusters | stim/integ clusters | other clusters | clusters holding half of them | held-out peak r (odd-passing) | within 10 ms (null) |
|---|---|---|---|---|---|---|---|---|
| paper | 6,931 (792) | 2,432 (4.5%) | 24 (1.0%) | 741 (30.5%) | 1,667 (68.5%) | 13 | 0.65 | 49% (18%) |
| type-corrected | 10,980 (1,416) | 6,856 (12.6%) | 149 (2.2%) | 1,734 (25.3%) | 4,973 (72.5%) | 16 | 0.79 | 60% (20%) |
| pooled | 12,823 (1,023) | 10,143 (18.6%) | 333 (3.3%) | 2,398 (23.6%) | 7,412 (73.1%) | 17 | 0.86 | 67% (20%) |

Null: the same criterion with shuffle draw 0 as data (peak test only). Class sizes: sequence 12,182 (22% of units), stim/integ 3,632 (7%), other 38,755. Held-out peak r and within 10 ms: units passing on odd trials, odd vs even peak (step 3); permutation p ≤ .001 in all three.

Clusters with the most cross-validated units (fraction of the cluster passing):

| version | top clusters |
|---|---|
| paper | 28 s/i (62%), 26 s/i (28%), 27 s/i (34%), 22 s/i (16%), 42 (16%), 21 (16%), 31 (19%), 56 (22%), 32 (14%), 24 s/i (9%) |
| type-corrected | 24 s/i (42%), 22 s/i (45%), 21 (45%), 28 s/i (76%), 26 s/i (56%), 25 s/i (44%), 42 (37%), 31 (45%), 27 s/i (59%), 20 (23%) |
| pooled | 24 s/i (64%), 22 s/i (64%), 21 (63%), 25 s/i (69%), 26 s/i (75%), 20 (37%), 31 (61%), 28 s/i (81%), 30 (52%), 23 s/i (47%) |

s/i = stim/integ cluster; others are "other" clusters. No sequence cluster is among the top 15 in any version.

![all units, clusters](figs/seq_detect/allcells_clusters.png)

![all units, held out](figs/seq_detect/allcells_step3.png)

- **The units the published procedure calls sequence cells are mostly not in the Rastermap sequence clusters.** Sequence clusters hold 22% of units but 1.0–3.3% of cross-validated passing units; every sequence cluster passes at 0–4.5%, below the overall rate (4.5–18.6%; dotted line in the cluster figure).
- **They concentrate in the stim/integ clusters (22–28) and the clusters next to them in the sort** (20–21, 29–32, 37–42): clusters 15–42 hold 59–70% of passing units; stim/integ clusters alone 24–31%. Other hotspots: 56–57 and 70–72.
- **The detected population tiles 0–150 ms reproducibly on held-out trials** (r = 0.65–0.86, rising from paper to pooled), with a pile-up of peaks in the last ~20 ms (27–30% of odd-passing units), mostly other-cluster units (grey in the scatters).

Open caveats: |Δpeak| not circular; late-window peaks not examined; cluster ids are specific to this Rastermap fit (FIG5_CLUSTERS.md).

### Over all six trial types

Added 2026-11-24. `seq_detect_allcells_6types.py`. Output: [figs/seq_detect/allcells_step3_6types.png](figs/seq_detect/allcells_step3_6types.png). Same units and sort as above (units passing on odd trials, sorted by the odd peak on the preferred type or the pooled set), shown on the six types of the six-type run, odd and even halves; rows min-max scaled over their 432 bins per half.

![all units, six types](figs/seq_detect/allcells_step3_6types.png)

- The peak order from one type (or the pooled set) carries over to every type, on held-out trials too: the diagonal is visible in all six types in all three versions, sharpest in the congruent types (LLL, RRR) and mistake, noisier in change_b and the incongruent types (fewer trials).

## Next steps (one at a time)

2. ~~Detection~~: done, see "Step 2".
3. ~~Held-out sequence figure~~: done, see "Step 3".
