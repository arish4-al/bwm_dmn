# Agent notes

## Python environment

Run all Python with the existing conda env `iblenv`. Do not create a new env, and do not use the base interpreter or the other local envs (`ibl_bwm`, `ibl_repro_ephys`, `bioplnn`, `python-general`).

```bash
/Users/ariliu/opt/anaconda3/envs/iblenv/bin/python script.py
```

`conda run -n iblenv python script.py` is equivalent when conda is on `PATH`.

## Figure 5 cluster selection

`fig5.py` selects sequence and stim/integ neurons from hand-picked Rastermap cluster ids. Read `FIG5_CLUSTERS.md` before changing those lists or running on a new or refit stack. It covers which clusters were picked and why, where the Rastermap caches live, and how to re-pick clusters when no cache exists.

## brainwidemap

`brainwidemap` is not installed in `iblenv`. Its functions (`bwm_query`, `load_good_units`, `load_trials_and_mask`, `bwm_units`) are only used when rebuilding datasets from raw insertions, in `dmn_bwm.py`, `dmn_ari.py`, `dmn_new.py`, and `granger.py`.

Cached analyses do not call it. That includes `regional_group`, `fig5.py`, and `seq_analysis.py`. Those modules still import `brainwidemap` at import time, so a missing package fails the import even when no function from it runs. If that happens, point `PYTHONPATH` at the checkout instead of installing the package into `iblenv`:

```bash
PYTHONPATH=/Users/ariliu/int-brain-lab/paper-brain-wide-map \
  /Users/ariliu/opt/anaconda3/envs/iblenv/bin/python script.py
```
