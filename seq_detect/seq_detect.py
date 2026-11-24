#!/usr/bin/env python3
"""Sequence-cell detection as in Taxidis et al. 2020 (Neuron) / Dorian et al.
2025 (Cell Reports), on every good unit of the stack insertions.

Per unit, per stimulus-aligned trial type (fig5.CONDITIONS, [0, 150] ms after
stimulus onset) and per trial set (all, odd, even trials by position in the
type's own list, as the stack):

  statistic   max over the 72 columns of the trial-summed PETH (12.5 ms
              boxcar counts; column i = j*6 + k starts at j*12.5 + k*2 ms,
              as the stack). Max of the sum = n_trials * max of the mean.
  null        2000 draws; in each, every trial's spike train is circularly
              shifted on its own within [0, 150) ms by a random offset,
              uniform over the 300 offsets of the 0.5 ms grid (the same as
              +-75 ms, with +75 and -75 counted once), the trials are summed and the max
              over columns taken. Columns are circular here (a column that
              runs past 150 ms wraps to 0), for the data too, so data and
              null differ only in the shifts. The stack's own columns (which
              run to 160 ms) are kept separately as P_exact.
  p           (1 + #{null max >= data max}) / (D + 1)
  p0          calibration: draw 0 treated as data, against draws 1..D-1
  p_corr      max over types handled: min over types of p, against the same
              min computed per draw (each draw's per-type p is its rank among
              that type's draws). p0_corr: the same for draw 0 (draw 0 is
              also in the per-type ranks it is compared with: slightly
              conservative).
  n_active    trials with >= 1 spike in [0, 150) ms (Taxidis pre-filter:
              >= max(10% of trials, 3))
  rel         fraction of trials with >= 1 spike within +-W of the peak
              (peak = centre of the max column of the peak set; circular),
              W in WIN_MS; for (eval set, peak set) in REL_PAIRS. Dorian's
              trial-reliability criterion is >= 20% within +-0.5 s of 3.5 s.

The shuffle of all trials is the sum of the odd and the even draw d (both
shift every trial independently), so all three sets cost two shuffles.

Inputs, disk only (no ONE, no Alyx), as contrast_split/:
  $ONE_CACHE_DIR/manifold/insertion_cache/{eid}_{probe}.npy   spikes
  $ONE_CACHE_DIR/bwm_tables/trials.pqt                        trials + masks
  $ONE_CACHE_DIR/bwm_tables/clusters.pqt                      uuid, cluster_id, label

  compute  one shard of insertions -> OUT/shards/{pid}.npz (skips existing)
           --check also bins with brainbox (stack code) and asserts P_exact
           equals it (local test)
  stack    all shards -> OUT/seq_detect.npz

  python seq_detect.py compute --shard 0 --n-shards 16 --out OUT
  python seq_detect.py stack --out OUT
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
sys.path.insert(0, str(HERE.parent / 'stint_thin'))
from stint_thin import SAT, STS, T_BIN, bin_peth, trial_types  # noqa: E402

INSERTIONS = HERE.parent / 'stint_thin' / 'data' / 'insertions.csv'
TYPES = ['block_change_s', 'stimLbLcL', 'stimLbRcL', 'stimRbRcR', 'stimRbLcR',
         'mistake_s']                                # fig5.CONDITIONS order
SETS = ['all', 'odd', 'even']
NB = 72
FINE = 0.0005                     # s; every column edge lies on this grid
NF = 300                          # fine bins in [0, 150) ms (circular world)
WIDTH = 25                        # column width in fine bins (12.5 ms)
STARTS = np.array([25 * j + 4 * k for j in range(12) for k in range(6)])   # column i = 6j + k
WIN_MS = (10, 20, 30)
REL_PAIRS = [('all', 'all'), ('odd', 'odd'), ('even', 'even'),
             ('even', 'odd'), ('odd', 'even')]    # (eval set, peak set)
N_DRAWS = 2000
CHUNK_ELEMS = 2 * 10 ** 7         # draws x cells x NF per bincount
CHUNK_SPIKES = int(os.environ.get('SEQ_DETECT_CHUNK_SPIKES', 10 ** 5))   # draws x spikes per chunk (small is faster: cache)


def session_events(trials_all, eid):
    trials = trials_all[trials_all['eid'] == eid].reset_index(drop=True)
    if trials.empty:
        raise RuntimeError(f'no trials for {eid} in trials.pqt')
    mask = trials['bwm_include'].to_numpy().astype(bool)
    for s in SAT:
        mask &= ~trials[s].to_numpy().astype(bool)
    tts = trial_types(trials, mask)
    out = {}
    for k in TYPES:
        col, m, pp = tts[k]
        if pp != (0, .15):
            raise RuntimeError(f'{k}: window {pp}, expected (0, 0.15)')
        out[k] = trials[col].to_numpy()[np.where(m)[0]]
    return out


def colsum(H):
    """(..., NF) fine counts -> (..., 72) circular column sums."""
    ext = np.concatenate([H, H[..., :WIDTH]], -1)
    cs = np.concatenate([np.zeros(H.shape[:-1] + (1,), ext.dtype), np.cumsum(ext, -1)], -1)
    return cs[..., STARTS + WIDTH] - cs[..., STARTS]


def gather_spikes(times, cidx, ev):
    """Spikes in [ev, ev + 160 ms) per trial: trial index, cell index, time
    relative to the event, and the exact absolute time (for the stack
    binning expression)."""
    lo = np.searchsorted(times, ev)
    hi = np.searchsorted(times, ev + 0.16)
    n = hi - lo
    tr = np.repeat(np.arange(len(ev)), n)
    idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if n.sum() else np.zeros(0, int)
    return tr, cidx[idx], times[idx]


def exact_cols(t_abs, ev_tr):
    """Stack column membership, with bin_spikes2D's own float expression:
    floor((t - (ev + k*STS)) / T_BIN). Returns (spike index, column)."""
    si, cols = [], []
    for k in range(int(T_BIN // STS)):
        j = np.floor((t_abs - (ev_tr + k * STS)) / T_BIN).astype(np.int64)
        ok = (j >= 0) & (j < 12)
        si.append(np.where(ok)[0]), cols.append(j[ok] * 6 + k)
    return np.concatenate(si), np.concatenate(cols)


def max_null(sp, n_c, rng_shifts, dc):
    """Generator over chunks of dc draws: (Dc, cells, NF) shifted fine
    counts, from spikes sp = (trial, cell, fine bin) and shifts (D, n_trials).
    Callers zipping two of these must pass the same dc."""
    tr, ci, fb = sp
    D = rng_shifts.shape[0]
    for a in range(0, D, dc):
        b = min(D, a + dc)
        sh = rng_shifts[a:b][:, tr]                         # (Dc, spikes)
        f = (fb[None] + sh) % NF
        ind = (np.arange(b - a)[:, None] * n_c + ci[None]) * NF + f
        H = np.bincount(ind.ravel(), minlength=(b - a) * n_c * NF)
        yield H.reshape(b - a, n_c, NF)


def per_type(times, cidx, n_c, ev, rng):
    """Everything for one trial type; see the module docstring."""
    ev = ev[np.isfinite(ev)]                                # placeholder trials, as stack_concat
    n = len(ev)
    sets = {'all': np.arange(n), 'odd': np.arange(0, n, 2), 'even': np.arange(1, n, 2)}
    ntr = np.array([len(sets[s]) for s in SETS])
    tr, ci, t_abs = gather_spikes(times, cidx, ev)
    rel_t = t_abs - ev[tr]
    # exact stack columns (non-circular, to 160 ms): odd/even means
    si, cols = exact_cols(t_abs, ev[tr])
    half = tr[si] % 2                                       # 0 = odd (train), 1 = even (test)
    E = np.bincount((half * n_c + ci[si]) * NB + cols, minlength=2 * n_c * NB).reshape(2, n_c, NB)
    with np.errstate(invalid='ignore', divide='ignore'):
        P_exact = (E / ntr[1:, None, None]).astype(np.float32)   # NaN if a half is empty
    # circular world
    w = rel_t < NF * FINE
    fb = np.minimum(np.floor(rel_t[w] / FINE).astype(np.int64), NF - 1)
    trw, ciw = tr[w], ci[w]
    F = np.bincount((trw * n_c + ciw) * NF + fb,
                    minlength=n * n_c * NF).reshape(n, n_c, NF).astype(np.int32)
    act = F.sum(-1) > 0                                     # (n, cells)
    S = {s: F[sets[s]].sum(0) for s in SETS}                # (cells, NF)
    C = {s: colsum(S[s]) for s in SETS}                     # (cells, 72)
    obs = np.stack([C[s].max(-1) for s in SETS], -1)        # (cells, 3)
    peak = np.stack([C[s].argmax(-1) for s in SETS], -1)
    n_active = np.stack([act[sets[s]].sum(0) for s in SETS], -1)
    P_circ = np.stack([C['odd'] / max(ntr[1], 1), C['even'] / max(ntr[2], 1)]).astype(np.float32)
    if ntr[1] == 0:
        P_circ[0] = np.nan
    if ntr[2] == 0:
        P_circ[1] = np.nan

    # trial reliability around the peak (circular windows)
    wmax = max(WIN_MS) * 2
    Fx = np.concatenate([F, F[..., :2 * wmax + WIDTH]], -1)
    cs = np.concatenate([np.zeros((n, n_c, 1), np.int32), np.cumsum(Fx, -1, dtype=np.int32)], -1)
    del Fx
    rel = np.zeros((n_c, len(REL_PAIRS), len(WIN_MS)), np.float32)
    cells = np.arange(n_c)
    for pi, (es, ps) in enumerate(REL_PAIRS):
        idx = sets[es]
        if len(idx) == 0:
            rel[:, pi] = np.nan
            continue
        centre = STARTS[peak[:, SETS.index(ps)]] + 12       # fine bin at the column centre
        for wi, wms in enumerate(WIN_MS):
            w2 = wms * 2
            a = (centre - w2) % NF                          # start, circular
            ti, cj = idx[:, None], cells[None, :]
            cnt = cs[ti, cj, (a + 2 * w2 + 1)[None]] - cs[ti, cj, a[None]]
            rel[:, pi, wi] = (cnt > 0).mean(0)

    # shuffles: odd and even draws, all = odd + even
    shifts = {s: rng.integers(0, NF, size=(N_DRAWS, n)) for s in ('odd', 'even')}
    M = np.zeros((N_DRAWS, n_c, 3), np.int64)
    sp = {}
    for s in ('odd', 'even'):
        m = np.isin(trw, sets[s])
        sp[s] = (trw[m], ciw[m], fb[m])
    nsp = max(len(sp['odd'][0]), len(sp['even'][0]), 1)
    dc = max(1, min(CHUNK_ELEMS // (n_c * NF), CHUNK_SPIKES // nsp))
    d0 = 0
    for Ho, He in zip(max_null(sp['odd'], n_c, shifts['odd'], dc),
                      max_null(sp['even'], n_c, shifts['even'], dc)):
        if Ho.shape[0] != He.shape[0]:
            raise RuntimeError('odd and even draw chunks out of step')
        dc_ = Ho.shape[0]
        M[d0:d0 + dc_, :, 1] = colsum(Ho).max(-1)
        M[d0:d0 + dc_, :, 2] = colsum(He).max(-1)
        M[d0:d0 + dc_, :, 0] = colsum(Ho + He).max(-1)
        d0 += dc_
    if d0 != N_DRAWS:
        raise RuntimeError(f'{d0} draws computed, expected {N_DRAWS}')
    p = (1 + (M >= obs[None]).sum(0)) / (N_DRAWS + 1)       # (cells, 3)
    p0 = (1 + (M[1:] >= M[:1]).sum(0)) / N_DRAWS
    for si_, s in enumerate(SETS):                          # an empty set tests nothing
        if ntr[si_] == 0:
            p[:, si_] = p0[:, si_] = 1.0
    # per-draw p of every draw within its own type and set (for p_corr)
    Ms = np.sort(M, 0)
    p_null = np.empty(M.shape, np.float32)
    for c in range(n_c):
        for s in range(3):
            p_null[:, c, s] = (N_DRAWS - np.searchsorted(Ms[:, c, s], M[:, c, s], 'left')) / N_DRAWS
    return dict(ntr=ntr, n_active=n_active, obs=obs, peak=peak, p=p, p0=p0,
                p_null=p_null, rel=rel, P_circ=P_circ, P_exact=P_exact)


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
          f'{N_DRAWS} draws', flush=True)

    cl = pd.read_parquet(args.clusters or one_dir / 'bwm_tables' / 'clusters.pqt',
                         columns=['uuids', 'cluster_id', 'pid', 'label'])
    cl = cl[(cl['label'] == 1) & cl['pid'].isin(mine)]
    trials_all = pd.read_parquet(args.trials or one_dir / 'bwm_tables' / 'trials.pqt')
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
        ids = units['cluster_id'].to_numpy().astype(np.int64)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError(f'{pid}: duplicate cluster ids')
        if not np.isin(ids, cache['clusters']['cluster_id']).all():
            raise RuntimeError(f'{pid}: cells missing from the insertion cache')
        clus = cache['clusters']['cluster_id'][cache['spikes']['clusters']]
        keep = np.isin(clus, ids)
        times, clus = cache['spikes']['times'][keep], clus[keep]
        del cache
        if np.any(np.diff(times) < 0):
            raise RuntimeError(f'{pid}: spike times not sorted')
        cidx = np.searchsorted(ids, clus)
        n_c = len(ids)
        t1 = time.time()
        events = session_events(trials_all, eid)
        seed = zlib.crc32(pid.encode())
        R = {}
        for ti, k in enumerate(TYPES):
            R[k] = per_type(times, cidx, n_c, events[k], np.random.default_rng([seed, ti]))
            if args.check:
                ev = events[k][np.isfinite(events[k])]
                X = bin_peth(times, clus, ids, ev, 0, .15)
                ref = np.stack([X[0::2].mean(0), X[1::2].mean(0)]).astype(np.float32)
                diff = np.nanmax(np.abs(ref - R[k]['P_exact']), initial=0)
                print(f'    check {k}: max |P_exact - bin_peth| = {diff:.2e}', flush=True)
                if not diff <= 1e-5:
                    raise RuntimeError(f'{pid} {k}: P_exact differs from bin_peth by {diff}')
        # p corrected for the max over types, per set
        PN = np.stack([R[k]['p_null'] for k in TYPES], -1)      # (D, cells, 3, 6)
        PO = np.stack([R[k]['p'] for k in TYPES], -1)           # (cells, 3, 6)
        ntr = np.stack([R[k]['ntr'] for k in TYPES], -1)        # (3, 6)
        PN = np.where(ntr[None, None] > 0, PN, 1.0)
        mn_null, mn_obs = PN.min(-1), PO.min(-1)                # (D, cells, 3), (cells, 3)
        p_corr = (1 + (mn_null <= mn_obs[None]).sum(0)) / (N_DRAWS + 1)
        p0_corr = (1 + (mn_null[1:] <= mn_null[:1]).sum(0)) / N_DRAWS

        def st(key, axis=1):
            return np.stack([R[k][key] for k in TYPES], axis)
        res = {'uuid': units['uuids'].to_numpy().astype(str), 'cluster_id': ids,
               'ntr': ntr.T,                                     # (6, 3)
               'n_active': st('n_active').astype(np.int32),     # (cells, 6, 3)
               'obs': st('obs').astype(np.int32),
               'peak': st('peak').astype(np.int16),
               'p': st('p').astype(np.float32), 'p0': st('p0').astype(np.float32),
               'p_corr': p_corr.astype(np.float32), 'p0_corr': p0_corr.astype(np.float32),
               'rel': st('rel'),                                 # (cells, 6, pairs, W)
               'P_circ': st('P_circ', 2),                        # (2, cells, 6, 72)
               'P_exact': st('P_exact', 2)}
        tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
        np.savez_compressed(tmp, **res)
        os.replace(tmp, f)
        print(f'[{i + 1}/{len(mine)}] {pid} {n_c} cells, {len(times)} spikes, '
              f'load {t1 - t0:.1f} s, detect {time.time() - t1:.1f} s', flush=True)


# ---------------------------------------------------------------- stack
def stack(args):
    pids = sorted(pd.read_csv(INSERTIONS)['pid'])
    if args.pids:
        pids = [p for p in pids if p in set(args.pids)]
    shards = {p: args.out / 'shards' / f'{p}.npz' for p in pids}
    missing = [p for p, f in shards.items() if not f.exists()]
    if missing:
        raise SystemExit(f'{len(missing)} insertions not computed, e.g. {missing[:3]}')
    keys_c = ['uuid', 'cluster_id', 'n_active', 'obs', 'peak', 'p', 'p0', 'p_corr',
              'p0_corr', 'rel']
    acc = {k: [] for k in keys_c + ['P_circ', 'P_exact', 'ntr', 'pid']}
    for p, f in shards.items():
        z = np.load(f)
        nc = len(z['uuid'])
        for k in keys_c:
            acc[k].append(z[k])
        acc['P_circ'].append(z['P_circ']), acc['P_exact'].append(z['P_exact'])
        acc['ntr'].append(np.broadcast_to(z['ntr'], (nc,) + z['ntr'].shape))
        acc['pid'].append(np.full(nc, p))
    out = {k: np.concatenate(v, 1 if k in ('P_circ', 'P_exact') else 0) for k, v in acc.items()}
    f = args.out / 'seq_detect.npz'
    np.savez_compressed(f, **out, types=np.array(TYPES), sets=np.array(SETS),
                        rel_pairs=np.array([f'{a}@{b}' for a, b in REL_PAIRS]),
                        win_ms=np.array(WIN_MS), n_draws=N_DRAWS,
                        starts_ms=STARTS * FINE * 1e3)
    print(f'{len(out["uuid"])} cells from {len(pids)} insertions -> {f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['compute', 'stack'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--clusters', type=Path, default=None,
                    help='clusters.pqt (default $ONE_CACHE_DIR/bwm_tables/clusters.pqt)')
    ap.add_argument('--trials', type=Path, default=None,
                    help='trials.pqt (default $ONE_CACHE_DIR/bwm_tables/trials.pqt)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=1)
    ap.add_argument('--pids', nargs='*', help='only these insertions (tests)')
    ap.add_argument('--one-cache-dir', type=Path, default=None)
    ap.add_argument('--check', action='store_true',
                    help='assert P_exact equals the stack binning (brainbox)')
    ap.add_argument('--force', action='store_true', help='recompute existing shards')
    args = ap.parse_args()
    compute(args) if args.step == 'compute' else stack(args)


if __name__ == '__main__':
    sys.exit(main())
