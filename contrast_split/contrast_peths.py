#!/usr/bin/env python3
"""Odd/even PETHs of every stack cell on the six stimulus-aligned trial types,
split by stimulus contrast.

Same binning and trial definitions as the concat stack (via stint_thin.py:
dmn_bwm.concat_PETHs + stack_concat, int-brain-lab/bwm_dmn main), but only
the six stimulus-aligned types of fig5.CONDITIONS, [0, 150] ms after stimulus
onset, 72 bins each (12.5 ms boxcar counts at a 2 ms stride). Each type's
trial list is further restricted to a contrast group, and odd/even trials
(by position in that restricted list, as the stack does per type) are
averaged into train (P0) and test (P1).

Contrast groups (|contrast| of the trial's stimulus, either side):
  all      every contrast: reproduces the stack's six segments exactly;
           only a check (costs no extra binning, 1/4 of the output)
  low      0, 0.0625, 0.125
  high     0.25, 1
  low_nz   0.0625, 0.125 (low without the zero-contrast trials)

Median trials per session (both halves) from bwm_tables/trials.pqt, low / high:
change_b 11 / 11, congruent L/R 60 / 66 and 59 / 57, incongruent L/R 11 / 14
and 10 / 14, mistake 54 / 5. High-contrast mistakes are rare.

Cells: every good unit (clusters.pqt label == 1) of the 515 stack insertions
(stint_thin/data/insertions.csv); no cell selection. Rows are identified by
uuid, so the stack's groups (sequence, stim/integ) and the exactness check of
group 'all' against concat_cvTrue.npy are done locally by joining on uuid.

Inputs, disk only (no ONE, no Alyx):
  $ONE_CACHE_DIR/manifold/insertion_cache/{eid}_{probe}.npy   spikes
  $ONE_CACHE_DIR/bwm_tables/trials.pqt                        trials + masks
  $ONE_CACHE_DIR/bwm_tables/clusters.pqt                      uuid, cluster_id, label

Each type's trials are binned once; every contrast group is a subset of
those binned trials, so spikes are binned once per type, not once per group.

  compute  one shard of insertions -> OUT/shards/{pid}.npz (skips existing)
  stack    all shards -> OUT/contrast_peths.npz: P0, P1 (cells, groups x 6
           types x 72 bins) float32, columns labelled by col_group/col_type;
           ntrials (cells, groups, 6, 2); uuid, pid, cluster_id

  python contrast_peths.py compute --shard 0 --n-shards 4 --out OUT
  python contrast_peths.py stack --out OUT
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stint_thin'))
from stint_thin import SAT, bin_peth, trial_types  # noqa: E402

INSERTIONS = HERE.parent / 'stint_thin' / 'data' / 'insertions.csv'
TYPES = ['block_change_s', 'stimLbLcL', 'stimLbRcL', 'stimRbRcR', 'stimRbLcR',
         'mistake_s']                                # fig5.CONDITIONS order
NB = 72
GROUPS = {'all': None,
          'low': (0.0, 0.0625, 0.125),
          'high': (0.25, 1.0),
          'low_nz': (0.0625, 0.125)}


def session_events(trials_all, eid):
    """{type: (event times, |contrast| per trial, (pre, post))}, the type's
    trials in session order."""
    trials = trials_all[trials_all['eid'] == eid].reset_index(drop=True)
    if trials.empty:
        raise RuntimeError(f'no trials for {eid} in trials.pqt')
    mask = trials['bwm_include'].to_numpy().astype(bool)
    for s in SAT:
        mask &= ~trials[s].to_numpy().astype(bool)
    tts = trial_types(trials, mask)
    c = (np.nan_to_num(trials['contrastLeft'].to_numpy())
         + np.nan_to_num(trials['contrastRight'].to_numpy()))
    out = {}
    for k in TYPES:
        col, m, pp = tts[k]
        i = np.where(m)[0]
        out[k] = (trials[col].to_numpy()[i], c[i], pp)
    return out


def oddeven_groups(times, clus, ids, ev, con, pre, post, groups):
    """Bin the type's trials once, then per contrast group: odd- and
    even-trial means (cells, 72) by position in the group's own trial list,
    and the two trial counts. A half with no trials gives NaN."""
    n_c = len(ids)
    nan = np.full((n_c, NB), np.nan, np.float32)
    if len(ev) == 0:
        return {g: (nan, nan, (0, 0)) for g in groups}
    X = bin_peth(times, clus, ids, ev, pre, post)      # (trials, cells, 72)
    if X.shape[2] != NB:
        raise RuntimeError(f'{X.shape[2]} bins, expected {NB}')
    ok = np.isfinite(X).any(axis=(1, 2))                # placeholder trials, as stack_concat
    X, con = X[ok], con[ok]
    out = {}
    for g in groups:
        Xg = X if GROUPS[g] is None else X[np.isin(con, GROUPS[g])]
        h = [x.mean(0).astype(np.float32) if len(x) else nan
             for x in (Xg[0::2], Xg[1::2])]
        out[g] = (h[0], h[1], (len(Xg[0::2]), len(Xg[1::2])))
    return out


# ---------------------------------------------------------------- compute
def compute(args):
    if 'ONE_CACHE_DIR' not in os.environ and args.one_cache_dir is None:
        raise SystemExit('set ONE_CACHE_DIR or --one-cache-dir')
    one_dir = Path(args.one_cache_dir or os.environ['ONE_CACHE_DIR'])
    cache_dir = one_dir / 'manifold' / 'insertion_cache'
    ins = pd.read_csv(INSERTIONS).set_index('pid')
    pids = sorted(ins.index)
    mine = pids[args.shard::args.n_shards]
    if args.pids:
        mine = [p for p in pids if p in set(args.pids)]
    out = args.out / 'shards'
    out.mkdir(parents=True, exist_ok=True)
    print(f'shard {args.shard}/{args.n_shards}: {len(mine)} of {len(pids)} insertions, '
          f'groups {args.groups}', flush=True)

    cl = pd.read_parquet(args.clusters or one_dir / 'bwm_tables' / 'clusters.pqt',
                         columns=['uuids', 'cluster_id', 'pid', 'label'])
    cl = cl[(cl['label'] == 1) & cl['pid'].isin(mine)]
    trials_all = pd.read_parquet(one_dir / 'bwm_tables' / 'trials.pqt')
    trials_all['eid'] = trials_all['eid'].astype(str)

    for i, pid in enumerate(mine):
        f = out / f'{pid}.npz'
        if f.exists() and not args.force:
            print(f'[{i + 1}/{len(mine)}] {pid} exists', flush=True)
            continue
        t0 = time.time()
        units = cl[cl['pid'] == pid].sort_values('cluster_id')
        if units.empty:
            raise RuntimeError(f'{pid}: no good units in clusters.pqt')
        eid, probe = str(ins.loc[pid, 'eid']), str(ins.loc[pid, 'probe'])
        cpath = cache_dir / f'{eid}_{probe}.npy'
        if not cpath.exists():
            raise FileNotFoundError(f'no insertion cache for {pid}: {cpath}')
        cache = np.load(cpath, allow_pickle=True).item()
        if str(cache['pid']) != pid:
            raise RuntimeError(f'{cpath.name} holds pid {cache["pid"]}, expected {pid}')

        ids = units['cluster_id'].to_numpy().astype(np.int64)   # sorted, as bin_spikes2D wants
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError(f'{pid}: duplicate cluster ids')
        if not np.isin(ids, cache['clusters']['cluster_id']).all():
            raise RuntimeError(f'{pid}: cells missing from the insertion cache')
        clus = cache['clusters']['cluster_id'][cache['spikes']['clusters']]
        keep = np.isin(clus, ids)
        times, clus = cache['spikes']['times'][keep], clus[keep]
        del cache
        t1 = time.time()
        events = session_events(trials_all, eid)
        seg = {k: oddeven_groups(times, clus, ids, ev, con, pre, post, args.groups)
               for k, (ev, con, (pre, post)) in events.items()}
        # columns: group-major, then type, then 72 bins
        P0 = np.concatenate([seg[k][g][0] for g in args.groups for k in TYPES], 1)
        P1 = np.concatenate([seg[k][g][1] for g in args.groups for k in TYPES], 1)
        nt = np.array([[seg[k][g][2] for k in TYPES] for g in args.groups])
        res = {'uuid': units['uuids'].to_numpy().astype(str), 'cluster_id': ids,
               'P0': P0, 'P1': P1, 'ntrials': nt, 'groups': np.array(args.groups)}
        tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
        np.savez_compressed(tmp, **res)
        os.replace(tmp, f)
        print(f'[{i + 1}/{len(mine)}] {pid} {len(ids)} cells, {len(times)} spikes, '
              f'load {t1 - t0:.1f} s, bin {time.time() - t1:.1f} s', flush=True)


# ---------------------------------------------------------------- stack
def stack(args):
    pids = sorted(pd.read_csv(INSERTIONS)['pid'])
    shards = {p: args.out / 'shards' / f'{p}.npz' for p in pids}
    missing = [p for p, f in shards.items() if not f.exists()]
    if missing:
        raise SystemExit(f'{len(missing)} insertions not computed, e.g. {missing[:3]}')
    P0, P1, nt, uuid, pid, cid, groups = [], [], [], [], [], [], None
    for p, f in shards.items():
        z = np.load(f)
        g = list(z['groups'])
        if groups is None:
            groups = g
        elif g != groups:
            raise SystemExit(f'{p}: groups {g}, expected {groups}')
        P0.append(z['P0']), P1.append(z['P1']), uuid.append(z['uuid'])
        cid.append(z['cluster_id']), pid.append(np.full(len(z['uuid']), p))
        nt.append(np.broadcast_to(z['ntrials'], (len(z['uuid']),) + z['ntrials'].shape))
    P0, P1, nt = np.concatenate(P0), np.concatenate(P1), np.concatenate(nt)
    col_group = np.repeat(groups, len(TYPES) * NB)
    col_type = np.tile(np.repeat(TYPES, NB), len(groups))
    f = args.out / 'contrast_peths.npz'
    np.savez_compressed(f, P0=P0, P1=P1, ntrials=nt, uuid=np.concatenate(uuid),
             pid=np.concatenate(pid), cluster_id=np.concatenate(cid),
             groups=np.array(groups), types=np.array(TYPES),
             col_group=col_group, col_type=col_type,
             contrasts=np.array([str(GROUPS[g]) for g in groups]))
    print(f'{len(P0)} cells from {len(pids)} insertions, P0/P1 {P0.shape} -> {f}')
    for i, g in enumerate(groups):
        print(f'  {g}: median trials per type (both halves) '
              f'{np.median(nt[:, i].sum(-1), axis=0).tolist()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['compute', 'stack'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clusters', type=Path, default=None,
                    help='clusters.pqt (default $ONE_CACHE_DIR/bwm_tables/clusters.pqt)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--pids', nargs='*', help='only these insertions (tests)')
    ap.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument('--one-cache-dir', type=Path, default=None)
    ap.add_argument('--force', action='store_true', help='recompute existing shards')
    args = ap.parse_args()
    compute(args) if args.step == 'compute' else stack(args)


if __name__ == '__main__':
    sys.exit(main())
