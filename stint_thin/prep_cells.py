#!/usr/bin/env python3
"""Local step: write the stim/integ cell list that the ORCD thinning job reads.

Output: stint_thin/data/stint_cells.npz (small, committed so ORCD needs no stack):

  row, pid, eid, probe, uuid, cluster_id   one entry per stim/integ cell, in
                                           stack (rastermap) order as fig5.py
  fr                                       stack ``fr`` (mean raw even half)
  chk_sumsq, chk_proj                      fingerprints of the stack rows, so
                                           the unthinned rebuild can be checked
                                           on ORCD without the 400 MB stack
  chk_vec                                  the random vector behind chk_proj
  variants, p_<name>                       per-cell spike keep probability

Variants (see journals/SEQ_STINT_NOISE.md, "Spike-thinned stim/integ cells"):
  p100         p = 1: no thinning; must reproduce the stack exactly
  rate_rank    each cell's rate goes to the sequence-cell fr at its fr rank
  rate_median  one p for all cells: median fr seq / median fr stim/integ

Run with the brainwidemap checkout on PYTHONPATH (fig5 imports it):
  PYTHONPATH=/Users/ariliu/int-brain-lab/paper-brain-wide-map \
    /Users/ariliu/opt/anaconda3/envs/iblenv/bin/python stint_thin/prep_cells.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fig5  # noqa: E402

CLUSTERS_PQT = Path.home() / 'Downloads/ONE/openalyx.internationalbrainlab.org/bwm_tables/clusters.pqt'


def rank_matched_p(fr_st, fr_seq):
    """Keep probability that moves each stim/integ cell to the sequence fr at
    the same quantile (capped at 1: never adds spikes)."""
    q = (np.argsort(np.argsort(fr_st)) + 0.5) / len(fr_st)
    return np.minimum(np.quantile(fr_seq, q) / fr_st, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stack', type=Path, default=fig5.NEW_DATA_PATH)
    ap.add_argument('--clusters', type=Path, default=CLUSTERS_PQT)
    ap.add_argument('--out', type=Path, default=HERE / 'data' / 'stint_cells.npz')
    args = ap.parse_args()

    raw = np.load(args.stack, allow_pickle=True).flat[0]
    isort = raw['isort']
    labels = raw['rm_labels'][isort]
    st_rows = isort[np.isin(labels, fig5.NEW_STINT_CLUSTERS)]
    seq_rows = isort[np.isin(labels, fig5.NEW_SEQ_CLUSTERS)]
    print(f'stim/integ {len(st_rows)}, sequence {len(seq_rows)}')

    pid = np.asarray(raw['pid']).astype(str)[st_rows]
    uuid = np.asarray(raw['uuids']).astype(str)[st_rows]

    ins = pd.read_csv(HERE / 'data' / 'insertions.csv').set_index('pid')
    missing = sorted(set(pid) - set(ins.index))
    if missing:
        raise RuntimeError(f'{len(missing)} pids not in insertions.csv: {missing[:3]}')
    eid = ins.loc[pid, 'eid'].to_numpy().astype(str)
    probe = ins.loc[pid, 'probe'].to_numpy().astype(str)

    cl = pd.read_parquet(args.clusters, columns=['uuids', 'cluster_id', 'pid'])
    cl = cl.drop_duplicates('uuids').set_index('uuids')
    cl_pid = cl.loc[uuid, 'pid'].to_numpy().astype(str)
    if not np.array_equal(cl_pid, pid):
        raise RuntimeError('uuid -> pid in clusters.pqt disagrees with the stack')
    cluster_id = cl.loc[uuid, 'cluster_id'].to_numpy().astype(np.int64)

    fr = raw['fr'][st_rows].astype(np.float64)
    fr_seq = raw['fr'][seq_rows].astype(np.float64)
    even = raw['concat'][st_rows].astype(np.float64)
    ztr = raw['concat_z_train'][st_rows].astype(np.float64)
    vec = np.random.default_rng(0).standard_normal(even.shape[1])

    variants = {
        'p100': np.ones(len(st_rows)),
        'rate_rank': rank_matched_p(fr, fr_seq),
        'rate_median': np.full(len(st_rows), np.median(fr_seq) / np.median(fr)),
    }
    for k, p in variants.items():
        print(f'{k:12s} p median {np.median(p):.3f} [{p.min():.3f}, {p.max():.3f}]'
              f'  thinned fr median {np.median(p * fr):.4f} (seq {np.median(fr_seq):.4f})')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, row=st_rows, pid=pid, eid=eid, probe=probe, uuid=uuid,
        cluster_id=cluster_id, fr=fr,
        chk_sumsq=(even ** 2).sum(1), chk_proj=zscore(ztr, axis=1) @ vec,
        chk_vec=vec, variants=np.array(list(variants)),
        **{f'p_{k}': v for k, v in variants.items()})
    print(f'{len(set(pid))} insertions -> {args.out}')


if __name__ == '__main__':
    main()
