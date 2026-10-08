# Figure 5: sequence and stim/integ cluster selection

`fig5.py` builds Figure 5 from Rastermap clusters that were picked by eye. This file records which clusters were picked for each dataset, why, and how to pick them again when the Rastermap sort changes.

## Datasets and Rastermap caches

| dataset | stack | Rastermap (`isort`, `rm_labels`) | output |
|---|---|---|---|
| old (`fig5.py`) | `dmn/res/concat_cvTrue_ephysFalse.npy` (53,021 cells) | `dmn/res/rm_concat_cvTrue_ephysFalse_nclusrm100_zsc{0,1}.npy`, written by `regional_group` (both give identical labels) | `dmn/figs/figure5/` |
| new (`fig5.py --data new`) | `/Users/ariliu/Downloads/concat_cvTrue.npy` (54,569 cells) | stored inside the stack itself (`isort`, `rm_labels`); no separate cache | `dmn/figs/figure5_new/` |

`dmn/` is `Path(one.cache_dir, 'dmn')`, currently `/Users/ariliu/Downloads/ONE/openalyx.internationalbrainlab.org/dmn`.

In both stacks, `concat_z_train` is the training half and `concat_z` is the held-out test half. In the new stack these are the odd and even trials (`cv_split='oddeven'`); `X_odd` equals `concat_z_train` and `X_even` equals `concat_z`. Rastermap was fit on `concat_z_train`.

`fig5.py` writes no Rastermap cache. The only Rastermap it runs (the re-sort of the shuffled sequence cells in panel b) is refit on every run, which takes about 20 s.

Cluster ids are only meaningful for the exact Rastermap fit they came from. If the stack or the Rastermap fit is regenerated, every list below has to be picked again.

## Old dataset (`SEQ_CLUSTERS`, `STINT_CLUSTERS`)

These come from the notebook (`bwm_dmn.ipynb`, cells "id sequence (and other) cells" and "examine compositions").

- Sequence: clusters 5-7, 11-20, 45-54 and 56-59 (11,539 cells).
- Stim/integ: cluster 4 (stim) plus clusters 38 and 1-3 (integ), 2,792 cells.
- Cells are selected as positions along `isort` between cluster boundaries, `[boundaries[c-1], boundaries[c])`, matching notebook cell 14.

## New dataset (`NEW_SEQ_CLUSTERS`, `NEW_STINT_CLUSTERS`)

`rm_labels` run from 0 to 99 and increase monotonically along `isort`, so cells are selected with `np.isin(rm_labels, clusters)` and kept in `isort` order.

### Sequence: 26 clusters, 12,182 cells

Each sequence is a diagonal in the train raster that is weak or absent in the test raster.

| sequence | clusters | notes |
|---|---|---|
| R_sR_cL_b (`stimRbLcR`) | 0-3, 5-7 | |
| L_sL_cR_b (`stimLbRcL`) | 44-49, 51-54 | 45-47 (and part of 54) carry the diagonal only in the movement-aligned copy, `sLbRchoiceL` |
| change_b (`block_change_s`) | 61-69 | the diagonal continues in `block_change_m` |

Excluded:

- Clusters 4 and 50 sit on a diagonal, but they are broad blocks that reproduce in the test half (train-test r ≈ 0.94), so they are not sequences.
- Clusters 58-60 are blobby and reproducible next to the change_b sequence.

### Stim/integ: clusters 22-28, 3,632 cells

These clusters respond in all six trial types with latency-graded responses that reproduce in the test half.

- Cluster 28 matches the old stim cluster: it holds 60% of the old stim cells.
- Clusters 22-27 match the old integ clusters: they hold 68% of the old integ cells.

Borderline, left out:

- Clusters 29-30 are a transient followed by suppression.
- Cluster 21 looks similar to 22 but contains almost no old stim/integ cells.
- Clusters 31-32 respond only in R_sR_cR_b and R_sR_cL_b, so they are trial-type specific.

### Cross-check against the old selection (matched by `uuids`, 53,004 shared cells)

| set | old cells overall | inside picked clusters |
|---|---|---|
| sequence | 22% | about 50% |
| stim/integ | 5% | 30-70% |

The diagnostic rasters used for the choice are in `figs/figure5_new/cluster_id/`, written by `plot_cluster_id`. Each shows train (top) and test (bottom) for all 21 trial types; picked cluster ids are coloured blue (sequence) or orange (stim/integ).

## How to pick clusters for a new Rastermap fit

1. Get `isort` and `rm_labels`.
   - If the stack already contains them, use those.
   - Otherwise, fit on the training half with the settings used for both datasets:

     ```python
     from rastermap import Rastermap
     model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75,
                       time_lag_window=5, bin_size=1).fit(raw['concat_z_train'])
     raw['isort'] = model.isort
     raw['rm_labels'] = model.embedding_clust   # take [:, 0] if 2D
     ```

   - Save the result next to the stack so the ids, and the picks made from them, stay fixed. `regional_group(..., rerun=True)` does the same thing for stacks under `dmn/res/` and writes `rm_<vers>_cv..._nclusrm100_zsc*.npy`.
   - Check that `rm_labels[isort]` is monotonic. If it is not, select cells by boundary positions, as for the old data, instead of by label.

2. Look at the rasters. `plot_cluster_id(raw, out_dir, ranges=...)` plots train and test for every trial type in `isort` order, with labelled cluster boundaries.
   - Start with wide ranges, for example `((0, 50), (50, 100))`, to find the structure.
   - Then zoom to ranges of about 15-20 clusters.

3. Pick the sequence clusters.
   - Look for a sharp diagonal in the train raster: each neuron peaks once, at a time that moves steadily from cell to cell. The diagonal usually appears in one or two trial types and in their movement-aligned copies.
   - The same rows should be weak or flat in the test raster, and the cluster's mean PETH should be roughly flat.
   - Leave out clusters on the diagonal that also show a broad block reproducing in the test half.

4. Pick the stim/integ clusters.
   - Look for stimulus-locked responses in all six stimulus trial types (`CONDITIONS`), with latency that varies smoothly across cells. Stim cells have an early sharp transient; integ cells have later or ramping responses.
   - The responses should reproduce clearly in the test raster.
   - Leave out clusters that respond in only some trial types.

5. Check the picks numerically.
   - The cluster-average train PETH and test PETH should be poorly correlated for sequence clusters and strongly correlated (r > 0.9) for stim/integ clusters.
   - If an earlier selection exists, match cells by `uuids` and check what fraction of each candidate cluster came from the old sequence and stim/integ sets.

6. Update the lists at the top of `fig5.py` and this file, then rerun `fig5.py` and look at panels a-c. Panel a train should show clean diagonals, and panel c should show consistent responses in every trial type.
