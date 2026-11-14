#!/usr/bin/env python3
"""Spike-thinned stim/integ cells: rebuild their odd/even PETHs from spikes.

Each spike of a stim/integ cell is kept with probability p (per cell, from
data/stint_cells.npz), once per seed, so both halves come from the same
thinned spike train. The thinned spikes are binned exactly as the concat
stack (dmn_bwm.concat_PETHs + stack_concat, int-brain-lab/bwm_dmn main):
21 trial types, 12.5 ms boxcar counts at a 2 ms stride, odd trials -> train
(P0), even trials -> test (P1), trial-averaged. Variant ``p100`` (no thinning)
must reproduce the stack exactly; ``stack`` checks it against fingerprints.

Inputs, disk only (no ONE, no Alyx):
  $ONE_CACHE_DIR/manifold/insertion_cache/{eid}_{probe}.npy   spikes
  $ONE_CACHE_DIR/bwm_tables/trials.pqt                        trials + masks
  data/stint_cells.npz                                        cells and p

  compute  one shard of insertions -> OUT/shards/{pid}.npz (skips existing)
  stack    all shards -> OUT/{variant}.npz with P0, P1 (seeds, cells, 1872)
           in stint_cells.npz order, after the p100 check

  python stint_thin.py compute --shard 0 --n-shards 8 --out OUT
  python stint_thin.py stack --out OUT
"""

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CELLS = HERE / 'data' / 'stint_cells.npz'

T_BIN, STS = 0.0125, 0.002
# stack type order (concat_cvTrue.npy 'ttypes'); 72 bins each except blockL,
# blockR, quiescence, fback1, fback0 (144): 1872 total
TTYPES = ['inter_trial', 'blockL', 'blockR', 'quiescence', 'block_change_s',
          'stimLbLcL', 'stimLbRcL', 'stimRbRcR', 'stimRbLcR', 'mistake_s',
          'motor_init', 'block_change_m', 'sLbLchoiceL', 'sLbRchoiceL',
          'sRbRchoiceR', 'sRbLchoiceR', 'mistake_m', 'choiceL', 'choiceR',
          'fback1', 'fback0']
N_BINS = 1872
SAT = ['saturation_stim_plus04', 'saturation_feedback_plus04',
       'saturation_move_minus02', 'saturation_stim_minus04_minus01',
       'saturation_stim_plus06', 'saturation_stim_minus06_plus06']


# ---------------------------------------------------------------- trial types
def deep_in_block(trials, pleft, depth=3):
    ar = np.arange(len(trials))[trials['probabilityLeft'] == pleft]
    ar_ = ar[trials['probabilityLeft'].to_numpy()[ar - depth] == pleft]
    out = np.zeros(len(trials), bool)
    out[ar_] = True
    return out


def first_three_after_block_switch(trials):
    pl = trials['probabilityLeft'].to_numpy()
    out = np.zeros(len(pl), bool)
    for b in np.where(np.diff(pl) != 0)[0] + 1:
        out[b:b + 3] = True
    return out


def trial_types(trials, mask):
    """align column, trial mask, (pre, post) per type, as concat_PETHs."""
    pl, ch, fb = trials['probabilityLeft'], trials['choice'], trials['feedbackType']
    cl = ~np.isnan(trials['contrastLeft'])
    cr = ~np.isnan(trials['contrastRight'])
    so = trials['stimOn_times'].to_numpy()
    idcs = [0] + list(np.where((so[1:] - trials['intervals_1'].to_numpy()[:-1]) > 1.15)[0] + 1)
    iti = np.isin(np.arange(len(so)), idcs)
    f3 = first_three_after_block_switch(trials)
    A = np.logical_and.reduce
    S, M, F = 'stimOn_times', 'firstMovement_times', 'feedback_times'
    return {
        'inter_trial': (S, A([mask, iti]), (1.15, -1)),
        'blockL': (S, A([mask, pl == .8]), (.4, -.1)),
        'blockR': (S, A([mask, pl == .2]), (.4, -.1)),
        'quiescence': (S, mask, (.4, -.1)),
        'block_change_s': (S, A([mask, f3]), (0, .15)),
        'stimLbLcL': (S, A([mask, cl, pl == .8, deep_in_block(trials, .8), ch == 1]), (0, .15)),
        'stimLbRcL': (S, A([mask, cl, pl == .2, deep_in_block(trials, .2), ch == 1]), (0, .15)),
        'stimRbRcR': (S, A([mask, cr, pl == .2, deep_in_block(trials, .2), ch == -1]), (0, .15)),
        'stimRbLcR': (S, A([mask, cr, pl == .8, deep_in_block(trials, .8), ch == -1]), (0, .15)),
        'mistake_s': (S, A([mask, fb == -1]), (0, .15)),
        'motor_init': (M, mask, (.15, 0)),
        'block_change_m': (M, A([mask, f3]), (.15, 0)),
        'sLbLchoiceL': (M, A([mask, cl, pl == .8, ch == 1]), (.15, 0)),
        'sLbRchoiceL': (M, A([mask, cl, pl == .2, ch == 1]), (.15, 0)),
        'sRbRchoiceR': (M, A([mask, cr, pl == .2, ch == -1]), (.15, 0)),
        'sRbLchoiceR': (M, A([mask, cr, pl == .8, ch == -1]), (.15, 0)),
        'mistake_m': (M, A([mask, fb == -1]), (.15, 0)),
        'choiceL': (M, A([mask, ch == 1]), (0, .15)),
        'choiceR': (M, A([mask, ch == -1]), (0, .15)),
        'fback1': (F, A([mask, fb == 1]), (0, .3)),
        'fback0': (F, A([mask, fb == -1]), (0, .3)),
    }


def session_events(trials_all, eid):
    """Per type: event times of the kept trials, in session order."""
    trials = trials_all[trials_all['eid'] == eid].reset_index(drop=True)
    if trials.empty:
        raise RuntimeError(f'no trials for {eid} in trials.pqt')
    mask = trials['bwm_include'].to_numpy().astype(bool)
    for s in SAT:
        mask &= ~trials[s].to_numpy().astype(bool)
    tts = trial_types(trials, mask)
    return {k: (trials[col].to_numpy()[np.where(m)[0]], pp)
            for k, (col, m, pp) in tts.items()}


# ---------------------------------------------------------------- binning
def bin_peth(times, clus, ids, ev, pre, post):
    """(trials, cells, bins) boxcar counts at 6 interleaved shifts, as
    concat_PETHs._bin_peth. ``ids`` sorted; spikes restricted to ``ids``."""
    from brainbox.singlecell import bin_spikes2D
    st = int(T_BIN // STS)
    bis = [bin_spikes2D(times, clus, ids, ev + k * STS, pre, post, T_BIN)[0]
           for k in range(st)]
    n, c, b = bis[0].shape
    ar = np.zeros((n, c, st * b), np.float32)
    for k in range(st):
        ar[:, :, k::st] = bis[k]
    return ar


def oddeven(times, clus, ids, events):
    """(cells, 1872) odd- and even-trial means, and trial counts per type."""
    P0, P1, n = [], [], []
    for k in TTYPES:
        ev, (pre, post) = events[k]
        X = bin_peth(times, clus, ids, ev, pre, post)
        nan_tr = ~np.isfinite(X).any(axis=(1, 2))   # placeholder trials, as stack_concat
        X = X[~nan_tr]
        P0.append(X[0::2].mean(0).astype(np.float32))
        P1.append(X[1::2].mean(0).astype(np.float32))
        n.append((len(X[0::2]), len(X[1::2])))
    P0, P1 = np.concatenate(P0, 1), np.concatenate(P1, 1)
    if P0.shape[1] != N_BINS:
        raise RuntimeError(f'{P0.shape[1]} bins, expected {N_BINS}')
    return P0, P1, np.array(n)


# ---------------------------------------------------------------- compute
def compute(args):
    if 'ONE_CACHE_DIR' not in os.environ and args.one_cache_dir is None:
        raise SystemExit('set ONE_CACHE_DIR or --one-cache-dir')
    one_dir = Path(args.one_cache_dir or os.environ['ONE_CACHE_DIR'])
    cache_dir = one_dir / 'manifold' / 'insertion_cache'
    C = np.load(args.cells)
    variants = [v for v in C['variants'] if v in args.variants]
    pids = sorted(set(C['pid']))
    mine = pids[args.shard::args.n_shards]
    if args.pids:
        mine = [p for p in pids if p in set(args.pids)]
    out = args.out / 'shards'
    out.mkdir(parents=True, exist_ok=True)
    print(f'shard {args.shard}/{args.n_shards}: {len(mine)} of {len(pids)} insertions, '
          f'variants {variants}, seeds {args.seeds}', flush=True)

    trials_all = pd.read_parquet(one_dir / 'bwm_tables' / 'trials.pqt')
    trials_all['eid'] = trials_all['eid'].astype(str)

    for i, pid in enumerate(mine):
        f = out / f'{pid}.npz'
        if f.exists() and not args.force:
            print(f'[{i + 1}/{len(mine)}] {pid} exists', flush=True)
            continue
        t0 = time.time()
        sel = np.where(C['pid'] == pid)[0]
        eid, probe = str(C['eid'][sel[0]]), str(C['probe'][sel[0]])
        cpath = cache_dir / f'{eid}_{probe}.npy'
        if not cpath.exists():
            raise FileNotFoundError(f'no insertion cache for {pid}: {cpath}')
        cache = np.load(cpath, allow_pickle=True).item()
        if str(cache['pid']) != pid:
            raise RuntimeError(f'{cpath.name} holds pid {cache["pid"]}, expected {pid}')

        cid = C['cluster_id'][sel]
        order = np.argsort(cid)                     # bin_spikes2D wants sorted ids
        ids = cid[order]
        clus = cache['clusters']['cluster_id'][cache['spikes']['clusters']]
        keep = np.isin(clus, ids)
        times, clus = cache['spikes']['times'][keep], clus[keep]
        if not np.isin(ids, cache['clusters']['cluster_id']).all():
            raise RuntimeError(f'{pid}: cells missing from the insertion cache')
        del cache
        events = session_events(trials_all, eid)
        spike_cell = np.searchsorted(ids, clus)     # cell index of every spike

        res = {'sel': sel}
        inv = np.argsort(order)                     # back to stint_cells order
        for v in variants:
            p = C[f'p_{v}'][sel][order]
            for s in ([0] if v == 'p100' else args.seeds):
                if v == 'p100':
                    t, c = times, clus
                else:
                    rng = np.random.default_rng([s, zlib.crc32(pid.encode())])
                    k = rng.random(len(times)) < p[spike_cell]
                    t, c = times[k], clus[k]
                P0, P1, n = oddeven(t, c, ids, events)
                res[f'{v}_s{s}_P0'] = P0[inv]
                res[f'{v}_s{s}_P1'] = P1[inv]
                res[f'{v}_s{s}_nspk'] = np.bincount(np.searchsorted(ids, c),
                                                    minlength=len(ids))[inv]
        res['ntrials'] = n                          # (types, halves); same for all
        tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
        np.savez_compressed(tmp, **res)
        os.replace(tmp, f)
        print(f'[{i + 1}/{len(mine)}] {pid} {len(sel)} cells, {len(times)} spikes, '
              f'{time.time() - t0:.1f} s', flush=True)


# ---------------------------------------------------------------- stack
def zrows(X):
    sd = X.std(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (X - X.mean(1, keepdims=True)) / sd


def stack(args):
    C = np.load(args.cells)
    n_cells = len(C['pid'])
    shards = {p: args.out / 'shards' / f'{p}.npz' for p in sorted(set(C['pid']))}
    missing = [p for p, f in shards.items() if not f.exists()]
    if missing:
        raise SystemExit(f'{len(missing)} insertions not computed, e.g. {missing[:3]}')

    first = np.load(next(iter(shards.values())))
    keys = sorted({k.rsplit('_', 1)[0] for k in first.files if k.endswith('_P0')})
    print('variant/seed keys:', keys)
    P = {k: (np.full((n_cells, N_BINS), np.nan, np.float32),
             np.full((n_cells, N_BINS), np.nan, np.float32),
             np.zeros(n_cells, np.int64)) for k in keys}
    ntrials = {}
    for pid, f in shards.items():
        z = np.load(f)
        sel = z['sel']
        ntrials[pid] = z['ntrials']
        for k in keys:
            P[k][0][sel], P[k][1][sel], P[k][2][sel] = z[f'{k}_P0'], z[f'{k}_P1'], z[f'{k}_nspk']

    # p100 must be the stack, bit for bit up to float32 rounding
    if 'p100_s0' in P:
        P0, P1 = (x.astype(np.float64) for x in P['p100_s0'][:2])
        d_fr = np.abs(P1.mean(1) - C['fr']).max()
        d_ss = np.abs((P1 ** 2).sum(1) - C['chk_sumsq']).max() / C['chk_sumsq'].max()
        d_pj = np.nanmax(np.abs(zrows(P0) @ C['chk_vec'] - C['chk_proj']))
        print(f'p100 check: max|fr| {d_fr:.2e}, rel max|sumsq| {d_ss:.2e}, '
              f'max|z(train)·v| {d_pj:.2e}')
        if d_fr > 1e-5 or d_ss > 1e-5 or d_pj > 1e-2:
            raise SystemExit('p100 rebuild does not match the stack: do not use the thinned sets')

    for v in sorted({k.rsplit('_s', 1)[0] for k in keys}):
        ks = sorted([k for k in keys if k.rsplit('_s', 1)[0] == v],
                    key=lambda k: int(k.rsplit('_s', 1)[1]))
        seeds = [int(k.rsplit('_s', 1)[1]) for k in ks]
        f = args.out / f'{v}.npz'
        np.savez(f, P0=np.stack([P[k][0] for k in ks]), P1=np.stack([P[k][1] for k in ks]),
                 nspk=np.stack([P[k][2] for k in ks]), seeds=seeds, row=C['row'],
                 p=C[f'p_{v}'], fr=C['fr'])
        print(f'{v}: seeds {seeds} -> {f}')
    np.save(args.out / 'ntrials.npy', ntrials, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['compute', 'stack'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=Path, default=CELLS)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--pids', nargs='*', help='only these insertions (tests)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--variants', nargs='+', default=['p100', 'rate_rank', 'rate_median'])
    ap.add_argument('--one-cache-dir', type=Path, default=None)
    ap.add_argument('--force', action='store_true', help='recompute existing shards')
    args = ap.parse_args()
    compute(args) if args.step == 'compute' else stack(args)


if __name__ == '__main__':
    sys.exit(main())
