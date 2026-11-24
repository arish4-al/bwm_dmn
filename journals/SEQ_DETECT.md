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
| time, memory | about 0.2 s per unit; 2.8 GB peak (234 units, with --check) |

The calibration is below 5%: spike counts are discrete, so maxima tie, and p counts ties against the data. The test is somewhat conservative, more so at low rates. Pass rates on these 700 units are not interpreted (6 insertions).

### Run on ORCD (the agent does not submit)

Commit and pull there first, then:

```bash
bash seq_detect/submit_seq_detect.sh
```

16 shards (2,850–4,290 units each, about 10–15 min) at 8 G and 2 h, then the stack job. Output: `$ONE_CACHE_DIR/dmn/res/seq_detect/seq_detect.npz` (about 50 MB); rsync it to `~/Downloads/seq_detect/`.

## Next steps (one at a time)

2. Detection: as in the paper (all trials) and cross-validated (peak, test and preferred type on odd trials; reliability and peak position on even). Pass rates per group (Rastermap sequence, stim/integ, fr+region partners, other).
3. Figure: passing units sorted by odd-trial peak, shown on even trials, with peak-time distributions; rate-matched partners alongside.
