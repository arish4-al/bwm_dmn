"""
Reconstruct Figure 5 (sequence vs stimulus/integrator neurons) as a single
multi-panel figure.

Panels follow the "id sequence (and other) cells" and "autocorrelation of
sequences" blocks of bwm_dmn.ipynb, using helpers from seq_analysis.py:

a  sequence neurons: train / test raster + trial-type average
b  sequence neuron shuffles: cells shuffled per trial type + rastermap
   re-sorted (train / test)
c  stimulus/integrator neurons: test raster + trial-type average
d  sequence neurons: bin x bin correlation within trial types (+ average)
e  stimulus/integrator neurons: same as d
f  population-average PETH (concatenated trial types, and trial-type average)
g  average firing rate per functional group
h  sequence neurons: bin x bin correlation across trial types, overall
   Pearson r between trial types, 2D cross-correlograms (ctxt seq similarity)
i  sequence neuron shuffles: bin x bin correlation across trial types
j  stimulus/integrator neurons: bin x bin correlation across trial types
k  regional composition of sequence and stim/integ neurons
l  fraction of cells per functional group

Run (iblenv):
    PYTHONPATH=../paper-brain-wide-map python fig5.py              # cached data
    PYTHONPATH=../paper-brain-wide-map python fig5.py --data new   # concat_cvTrue.npy

The new dataset (odd/even trial split, its own rastermap) has its own sequence
and stim/integ cluster ids, picked by eye from the rasters saved in
figs/figure5_new/cluster_id/; see FIG5_CLUSTERS.md. Panels g and l are skipped
for it.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import PowerNorm, SymLogNorm
import iblatlas
from rastermap import Rastermap

from dmn_bwm import (regional_group, plot_rm_cluster_profile, pth_dmn, br,
                     pal, peth_dictm)
from dmn_ari import sum_for_key
from seq_analysis import (
    bin_autocorr_mats,
    concat_bin_autocorr,
    overall_corr_matrix,
    _corr2d,
)

mpl.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

SEQ_COL = '#01789a'
STINT_COL = '#e58324'
SHUF_COL = '#888a8c'

CONDITIONS = [
    'block_change_s',
    'stimLbLcL',
    'stimLbRcL',
    'stimRbRcR',
    'stimRbLcR',
    'mistake_s',
]

SEQ_CLUSTERS = (list(range(5, 8)) + list(range(11, 21))
                + list(range(45, 55)) + list(range(56, 60)))
STINT_CLUSTERS = [38] + list(range(1, 5))

# new dataset (concat_cvTrue.npy), picked by eye for its stored rastermap fit;
# see FIG5_CLUSTERS.md for the rationale and how to re-pick after a refit.
NEW_DATA_PATH = Path(__file__).parent / 'concat_cvTrue.npy'
NEW_SEQ_CLUSTERS = ([0, 1, 2, 3, 5, 6, 7]
                    + [44, 45, 46, 47, 48, 49, 51, 52, 53, 54]
                    + list(range(61, 70)))
NEW_STINT_CLUSTERS = list(range(22, 29))
# (n cells, n sequence cells, n stim/integ cells) for the fit the lists match
NEW_EXPECTED_COUNTS = (54569, 12182, 3632)

# functional groups (rastermap cluster ids), as in "examine compositions"
GROUPS = [
    ('sequences', SEQ_CLUSTERS),
    ('stim', [4]),
    ('integ', [38] + list(range(1, 4))),
    ('move', list(range(72, 76))),
    ('move init', list(range(34, 38)) + list(range(41, 43)) + [10, 32, 39, 61]),
    ('move\n& fback', [0, 62, 63, 67, 68] + list(range(70, 72)) + [76, 77]
     + list(range(83, 89)) + [90, 96]),
    ('fback', list(range(64, 67)) + list(range(78, 83)) + [89, 94, 95, 97]),
]
GROUP_BAR_LABELS = ['seq', 'stim', 'integ', 'move', 'move init',
                    'mv & fb', 'fback', 'other']

RASTER_NORM = PowerNorm(gamma=0.5, vmin=0, vmax=1.5)
CORR_SYMLOG = dict(linthresh=0.05, vmin=-1, vmax=1, base=10)
CROSS_SMOOTH = (1, 2)
# keep every Nth cell-lag row of the 2D correlograms; with nearest-neighbour
# rendering this matches the full-resolution plot (block-averaging would blur
# the fine streak texture)
CROSS_ROW_STRIDE = 8


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data():
    r = regional_group('rm', vers='concat', ephys=False, nclus=100,
                       rerun=False, cv=True, grid_upsample=0, locality=0.75,
                       time_lag_window=5, symmetric=False, zsc=False)

    isort = r['isort']
    rc = np.asarray(r['acs'])[isort]
    boundaries = np.where(rc[1:] != rc[:-1])[0]

    def _rows(clusters):
        return np.concatenate([np.arange(boundaries[c - 1], boundaries[c])
                               for c in clusters])

    seq_rows, stint_rows = _rows(SEQ_CLUSTERS), _rows(STINT_CLUSTERS)
    z, z_tr = r['concat_z'][isort], r['concat_z_train'][isort]

    return dict(
        r=r,
        seq_peth=z[seq_rows], seq_tr_peth=z_tr[seq_rows],
        stint_peth=z[stint_rows], stint_tr_peth=z_tr[stint_rows],
    )


def load_new_data(path=NEW_DATA_PATH):
    """Load the new stack (concat_z_train = odd trials, concat_z = even
    trials); cells are taken in rastermap order."""
    raw = np.load(path, allow_pickle=True).flat[0]
    isort = raw['isort']
    labels_sorted = raw['rm_labels'][isort]
    seq_rows = isort[np.isin(labels_sorted, NEW_SEQ_CLUSTERS)]
    stint_rows = isort[np.isin(labels_sorted, NEW_STINT_CLUSTERS)]
    counts = (len(isort), len(seq_rows), len(stint_rows))
    if counts != NEW_EXPECTED_COUNTS:
        warnings.warn(
            f'cell counts {counts} != {NEW_EXPECTED_COUNTS}: {path.name} or its '
            'rastermap fit changed, so NEW_SEQ_CLUSTERS / NEW_STINT_CLUSTERS '
            'probably need re-picking (see FIG5_CLUSTERS.md)')

    acs = np.array(br.id2acronym(raw['ids'], mapping='Beryl'))
    r = dict(
        len=raw['len'],
        peth_dict={k: peth_dictm[k] for k in raw['ttypes']},
        Beryl=acs,
        cols=np.array([pal[a] for a in acs]),
    )
    return dict(
        r=r, raw=raw,
        seq_peth=raw['concat_z'][seq_rows],
        seq_tr_peth=raw['concat_z_train'][seq_rows],
        stint_peth=raw['concat_z'][stint_rows],
        stint_tr_peth=raw['concat_z_train'][stint_rows],
        seq_mask=np.isin(raw['rm_labels'], NEW_SEQ_CLUSTERS),
        stint_mask=np.isin(raw['rm_labels'], NEW_STINT_CLUSTERS),
    )


def plot_cluster_id(raw, out_dir, ranges=((0, 14), (16, 36), (40, 58), (56, 72))):
    """Train / test rasters (all trial types, rastermap order) used to pick
    sequence and stim/integ clusters; selected cluster ids are coloured."""
    out_dir.mkdir(parents=True, exist_ok=True)
    isort, lens = raw['isort'], raw['len']
    ls = raw['rm_labels'][isort]
    edges = np.concatenate([[0], np.flatnonzero(ls[1:] != ls[:-1]) + 1, [len(ls)]])
    seg_edges = np.cumsum(list(lens.values()))
    for lo, hi in ranges:
        r0, r1 = edges[lo], edges[hi]
        fig, axs = plt.subplots(2, 1, figsize=(24, 16), sharex=True)
        for ax, key in zip(axs, ['concat_z_train', 'concat_z']):
            img = raw[key][isort[r0:r1]]
            ax.imshow(img, cmap='gray_r', norm=RASTER_NORM, aspect='auto',
                      extent=(0, img.shape[1], r1, r0))
            for b in seg_edges[:-1]:
                ax.axvline(b, color='r', lw=0.8)
            for c in range(lo + 1, hi):
                ax.axhline(edges[c], color='tab:blue', lw=0.5)
            mids = (edges[lo:hi] + edges[lo + 1:hi + 1]) / 2
            ax.set_yticks(mids, [str(c) for c in range(lo, hi)], fontsize=10)
            for t, c in zip(ax.get_yticklabels(), range(lo, hi)):
                if c in NEW_SEQ_CLUSTERS:
                    t.set_color(SEQ_COL)
                    t.set_fontweight('bold')
                elif c in NEW_STINT_CLUSTERS:
                    t.set_color(STINT_COL)
                    t.set_fontweight('bold')
            ax.set_ylabel(f'{key}  (rm cluster)')
        axs[1].set_xticks(seg_edges - np.array(list(lens.values())) / 2,
                          list(lens), rotation=60, ha='right', fontsize=8)
        fig.suptitle(f'rastermap clusters {lo}-{hi - 1} '
                     '(blue = sequence, orange = stim/integ)')
        fig.tight_layout()
        fig.savefig(out_dir / f'rm_clusters_{lo}_{hi - 1}.png', dpi=100)
        plt.close(fig)


def condition_columns(lens, conditions):
    cols = []
    for key in conditions:
        cols.extend(range(sum_for_key(lens, key),
                          sum_for_key(lens, key, after=True)))
    return np.asarray(cols)


def subset_lens(lens, conditions):
    return {k: lens[k] for k in lens if k in conditions}


def condition_average(peth, seg_lens):
    return np.mean(np.stack(np.split(peth, np.cumsum(list(seg_lens.values()))[:-1],
                                     axis=1)), axis=0)


def shuffle_and_resort(peth, tr_peth, seg_lens, seed=0):
    """Shuffle cell identities independently within each trial type, then
    re-sort rows with Rastermap fitted on the train half (same as
    plot_raster_subset with shuffle=True, rastermap_sort=True)."""
    rng = np.random.default_rng(seed)
    peth, tr_peth = peth.copy(), tr_peth.copy()
    h = 0
    for seg_len in seg_lens.values():
        sl = slice(h, h + seg_len)
        perm = rng.permutation(peth.shape[0])
        peth[:, sl] = peth[perm, sl]
        tr_peth[:, sl] = tr_peth[perm, sl]
        h += seg_len

    X = np.nan_to_num(np.asarray(tr_peth, dtype=np.float32))
    row_var = np.var(X, axis=1)
    igood = np.isfinite(row_var) & (row_var > 0)
    Xg = X[igood]
    Xg = (Xg - Xg.mean(0, keepdims=True)) / (Xg.std(0, keepdims=True) + 1e-6)
    Xg = np.nan_to_num(Xg)

    max_pcs = min(tr_peth.shape) - 1
    model = Rastermap(n_PCs=min(200, max(1, max_pcs)),
                      n_clusters=min(100, tr_peth.shape[0]),
                      locality=0.75, time_lag_window=5, bin_size=1).fit(Xg)
    good_idx, bad_idx = np.flatnonzero(igood), np.flatnonzero(~igood)
    isort = np.concatenate([good_idx[np.asarray(model.isort, int)], bad_idx])
    return peth[isort], tr_peth[isort]


def group_stats(r):
    acs = np.asarray(r['acs']).astype(int)
    X = np.asarray(r['concat'])
    named = [c for _, cl in GROUPS for c in cl] + STINT_CLUSTERS
    other = sorted(set(range(100)) - set(named))
    names, n_cells, avg_fr = [], [], []
    for name, clusters in GROUPS + [('other', other)]:
        mask = np.isin(acs, clusters)
        names.append(name)
        n_cells.append(mask.sum())
        avg_fr.append(X[mask].mean())
    return names, np.array(n_cells), np.array(avg_fr)


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def add_ax(fig, x0, y0, w, h, **kw):
    """Axes from a top-left anchored rectangle in figure fractions."""
    return fig.add_axes([x0, 1 - y0 - h, w, h], **kw)


def panel_letter(fig, x, y, letter):
    """Draw a panel letter; artists added until the next letter belong to it."""
    if not hasattr(fig, '_panel_marks'):
        fig._panel_marks = []
    fig._panel_marks.append((letter, len(fig.axes), len(fig.texts)))
    fig.text(x, 1 - y, letter, fontsize=22, va='top', ha='left')


def save_panels(fig, out_dir, exts=('pdf', 'png'), pad=0.05, dpi=300):
    """Save each lettered panel as its own file, cropped from the full figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    marks = fig._panel_marks + [(None, len(fig.axes), len(fig.texts))]
    everything = fig.axes + fig.texts
    panels = []
    for (letter, a0, t0), (_, a1, t1) in zip(marks[:-1], marks[1:]):
        artists = fig.axes[a0:a1] + fig.texts[t0:t1]
        bbox = mpl.transforms.Bbox.union(
            [a.get_tightbbox(renderer) for a in artists])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).padded(pad)
        panels.append((letter, artists, bbox))
    for letter, artists, bbox in panels:
        keep = set(map(id, artists))
        for a in everything:
            a.set_visible(id(a) in keep)
        for ext in exts:
            fig.savefig(out_dir / f'fig5{letter}.{ext}', bbox_inches=bbox,
                        dpi=dpi)
    for a in everything:
        a.set_visible(True)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)


def raster(ax, img, seg_lens, labels=None, grayscale='0.4'):
    ax.set_facecolor(grayscale)
    ax.imshow(img, cmap='gray_r', norm=RASTER_NORM, aspect='auto', alpha=0.95)
    h = 0
    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for seg, seg_len in seg_lens.items():
        ax.axvline(h + seg_len, ls='--', lw=1, color='grey')
        if labels is not None:
            ax.text(h + seg_len / 2, 1.02, labels.get(seg, seg), rotation=60,
                    ha='left', va='bottom', rotation_mode='anchor',
                    fontsize=6.5, transform=trans, clip_on=False)
        h += seg_len
    ax.set_xlim(0, img.shape[1])
    ax.set_xticks([])


def peth_ylim(ax, default=(-0.3, 0.5), pad=0.05):
    """Notebook PETH ylim (mean_ylim / cond_avg_ylim), widened if a trace
    would be clipped."""
    ys = np.concatenate([line.get_ydata() for line in ax.get_lines()
                         if len(line.get_ydata()) > 2])
    return min(default[0], ys.min() - pad), max(default[1], ys.max() + pad)


def avg_raster(ax, img, title='avg\n(trial-\ntypes)'):
    ax.imshow(img, cmap='gray_r', norm=RASTER_NORM, aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, loc='left')


def bin_corr_grid(fig, rect, mats, conditions, labels, avg_rect):
    x0, y0, w, h = rect
    gap = 0.004
    cw, ch = (w - 2 * gap) / 3, (h - gap) / 2
    for idx, key in enumerate(conditions):
        i, j = divmod(idx, 3)
        ax = add_ax(fig, x0 + j * (cw + gap), y0 + i * (ch + gap), cw, ch)
        mat = mats[key].copy()
        np.fill_diagonal(mat, 0)
        ax.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm')
        clean(ax)
        if i == 0:
            ax.set_title(labels[key], fontsize=6.5, pad=2)
        else:
            ax.set_xlabel(labels[key], fontsize=6.5, labelpad=2)
        if idx == 0:
            ax.set_ylabel('time', fontsize=7)
    ax = add_ax(fig, *avg_rect)
    avg = mats['avg_conditions'].copy()
    np.fill_diagonal(avg, 0)
    ax.imshow(avg, vmin=-1, vmax=1, cmap='coolwarm')
    clean(ax)
    ax.set_title('average\n(trial-types)', fontsize=8)


def concat_corr(ax, corr, seg_lens, labels, ylabels=True, xlabels=True):
    corr = corr.copy()
    np.fill_diagonal(corr, 0)
    im = ax.imshow(corr, cmap='coolwarm', norm=SymLogNorm(**CORR_SYMLOG))
    edges = np.concatenate([[0], np.cumsum(list(seg_lens.values()))])
    for b in edges[1:-1]:
        ax.axvline(b - 0.5, color='k', lw=0.5, alpha=0.4)
        ax.axhline(b - 0.5, color='k', lw=0.5, alpha=0.4)
    centers = (edges[:-1] + edges[1:]) / 2
    names = [labels[k] for k in seg_lens]
    ax.set_xticks(centers)
    ax.set_xticklabels(names if xlabels else [], rotation=60, ha='right',
                       fontsize=6.5)
    ax.set_yticks(centers)
    ax.set_yticklabels(names if ylabels else [], fontsize=6.5)
    for s in ax.spines.values():
        s.set_visible(True)
    return im


def cross_corr_grid(fig, rect, peth, lens, conditions, labels):
    """2D cross-correlograms between trial types (image_crosscorrs with
    per-condition normalization, smoothed, symlog colour scale)."""
    segs = {k: peth[:, sum_for_key(lens, k):sum_for_key(lens, k, after=True)]
            for k in conditions}
    x0, y0, w, h = rect
    n = len(conditions)
    gap = 0.002
    cw, ch = (w - (n - 1) * gap) / n, (h - (n - 1) * gap) / n
    norm = SymLogNorm(linthresh=1e-2, base=10)
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            corr = _corr2d(segs[a], segs[b], use_fft=True,
                           smooth_sigma=CROSS_SMOOTH)
            center = corr[corr.shape[0] // 2, corr.shape[1] // 2]
            if center != 0:
                corr = corr / center
            # shared norm takes its range from the first full-resolution image,
            # as in plot_crosscorr_matrix_grid
            norm.autoscale_None(corr)
            corr = corr[::CROSS_ROW_STRIDE].astype(np.float32)
            ax = add_ax(fig, x0 + j * (cw + gap), y0 + i * (ch + gap), cw, ch)
            ax.imshow(corr, cmap='coolwarm', norm=norm, aspect='auto',
                      interpolation='nearest')
            clean(ax)
            if j == 0:
                ax.set_ylabel(labels[a], rotation=0, ha='right', va='center',
                              fontsize=6.5)
            if i == n - 1:
                ax.set_xlabel(labels[b], rotation=60, ha='right', fontsize=6.5)
            if i == 0 and j == n - 1:
                ax.set_title('time lag', fontsize=7, loc='right')
                ax.yaxis.set_label_position('right')
                ax.set_ylabel('cell lag', fontsize=7, rotation=-90,
                              va='bottom')


def region_pie(ax, acs, cols, mask, top_wedge_labels=15, min_region_count=50):
    """Polar region-composition wedges, same as the left panel of
    plot_rm_cluster_profile (root/void removed, fractions normalized by
    global region counts, canonical Beryl order, top-N wedges labelled)."""
    acs = np.asarray(acs)
    keep = ~np.isin(np.char.lower(acs.astype(str)), ['root', 'void'])
    acs, cols, mask = acs[keep], np.asarray(cols)[keep], np.asarray(mask)[keep]

    counts = Counter(acs[mask])
    global_counts = Counter(acs)
    reg2col = {}
    for a, c in zip(acs, cols):
        reg2col.setdefault(a, c)

    regs = [a for a in counts if global_counts[a] >= min_region_count]
    vals = np.array([counts[a] / global_counts[a] for a in regs])
    frac = vals / (vals.sum() + 1e-12)

    canonical = list(br.id2acronym(
        np.load(Path(iblatlas.__file__).parent / 'beryl.npy'), mapping='Beryl'))
    reg_sorted = [a for a in canonical if a in set(regs)]
    reg_sorted += sorted([a for a in regs if a not in set(reg_sorted)], key=str)
    idx = {a: i for i, a in enumerate(regs)}
    frac_sorted = np.array([frac[idx[a]] for a in reg_sorted])
    frac_sorted /= frac_sorted.sum() + 1e-12
    cols_sorted = [reg2col[a] for a in reg_sorted]

    theta_edges = np.concatenate(([0.0], 2 * np.pi * np.cumsum(frac_sorted)))
    theta, widths = theta_edges[:-1], np.diff(theta_edges)
    ax.bar(theta, np.ones_like(theta), width=widths, align='edge',
           color=cols_sorted, edgecolor='none')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)

    w_norm = widths / (2 * np.pi + 1e-12)
    w0, w1 = float(w_norm.min()), float(w_norm.max())
    denom = (w1 - w0) if (w1 - w0) > 1e-12 else 1.0
    for i in np.argsort(-frac_sorted)[:min(top_wedge_labels, len(reg_sorted))]:
        mid = theta[i] + 0.5 * widths[i]
        ang = np.degrees(mid)
        rot, ha = (ang + 180, 'right') if 90 < ang < 270 else (ang, 'left')
        t = float(np.clip((w_norm[i] - w0) / denom, 0, 1))
        ax.text(mid, 1.04, str(reg_sorted[i]), rotation=rot,
                rotation_mode='anchor', ha=ha, va='center',
                fontsize=3.0 + 12.0 * t, color=cols_sorted[i], clip_on=False,
                zorder=10)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(d, out_dir=None, composition=True):
    """Build the figure; panels g and l (functional-group composition) need
    the old cluster groups and are drawn only if ``composition``."""
    r = d['r']
    labels = r['peth_dict']
    lens = r['len']
    seg_lens = subset_lens(lens, CONDITIONS)
    cols = condition_columns(lens, CONDITIONS)

    seq, seq_tr = d['seq_peth'][:, cols], d['seq_tr_peth'][:, cols]
    stint = d['stint_peth'][:, cols]

    print('shuffling + rastermap re-sorting sequence neurons ...')
    shuf, shuf_tr = shuffle_and_resort(seq, seq_tr, seg_lens)

    print('correlations ...')
    bin_seq = bin_autocorr_mats(seq, seg_lens, CONDITIONS)
    bin_stint = bin_autocorr_mats(stint, seg_lens, CONDITIONS)
    cc_seq = concat_bin_autocorr(seq, seg_lens, CONDITIONS)
    cc_shuf = concat_bin_autocorr(shuf, seg_lens, CONDITIONS)
    cc_stint = concat_bin_autocorr(stint, seg_lens, CONDITIONS)
    overall_seq = overall_corr_matrix(seq, seg_lens, CONDITIONS)

    if composition:
        names, n_cells, avg_fr = group_stats(r)

    fig = plt.figure(figsize=(12, 14))

    # ---- a: sequence neurons rasters ----
    panel_letter(fig, 0.0, 0.0, 'a')
    fig.text(0.24, 0.995, 'sequence neurons', color=SEQ_COL, fontsize=14,
             ha='center', va='top')
    ax_tr = add_ax(fig, 0.045, 0.082, 0.155, 0.10)
    raster(ax_tr, seq_tr, seg_lens, labels)
    ax_tr.set_ylabel('cells')
    ax_tr.set_yticks([0, 10000], ['0', r'$10^4$'])
    ax_te = add_ax(fig, 0.205, 0.082, 0.155, 0.10, sharey=ax_tr)
    raster(ax_te, seq, seg_lens, labels)
    ax_te.tick_params(labelleft=False)
    fig.text(0.122, 0.968, 'train', color=SEQ_COL, fontsize=10, ha='center')
    fig.text(0.282, 0.968, 'test', color=SEQ_COL, fontsize=10, ha='center')
    avg_raster(add_ax(fig, 0.395, 0.082, 0.04, 0.10),
               condition_average(seq, seg_lens))

    # ---- b: sequence neuron shuffles ----
    panel_letter(fig, 0.455, 0.0, 'b')
    fig.text(0.59, 0.995, 'sequence neuron shuffles', color=SHUF_COL,
             fontsize=12, ha='center', va='top')
    fig.text(0.59, 0.968, 'train', color=SHUF_COL, fontsize=10, ha='center')
    ax = add_ax(fig, 0.495, 0.082, 0.185, 0.11)
    raster(ax, shuf_tr, seg_lens, labels)
    ax.set_yticks([])
    fig.text(0.59, 0.792, 'test', color=SHUF_COL, fontsize=10, ha='center')
    ax = add_ax(fig, 0.495, 0.212, 0.185, 0.11)
    raster(ax, shuf, seg_lens)
    ax.set_yticks([])

    # ---- c: stimulus/integrator neurons rasters ----
    panel_letter(fig, 0.705, 0.02, 'c')
    fig.text(0.86, 0.995, 'stimulus/integrator neurons', color=STINT_COL,
             fontsize=12, ha='center', va='top')
    fig.text(0.825, 0.96, 'test', fontsize=10, ha='center')
    ax = add_ax(fig, 0.735, 0.09, 0.18, 0.035)
    raster(ax, stint, seg_lens, labels)
    ax.set_yticks([])
    avg_raster(add_ax(fig, 0.93, 0.09, 0.03, 0.035),
               condition_average(stint, seg_lens))

    # ---- d / e: bin x bin correlation within trial types ----
    panel_letter(fig, 0.0, 0.2, 'd')
    fig.text(0.035, 1 - 0.27, 'correlns within\ntrial-types', rotation=90,
             fontsize=11, ha='center', va='center')
    bin_corr_grid(fig, (0.085, 0.215, 0.19, 0.115), bin_seq, CONDITIONS,
                  labels, avg_rect=(0.29, 0.245, 0.065, 0.065 * 12 / 14))

    panel_letter(fig, 0.705, 0.15, 'e')
    bin_corr_grid(fig, (0.725, 0.17, 0.19, 0.115), bin_stint, CONDITIONS,
                  labels, avg_rect=(0.925, 0.2, 0.065, 0.065 * 12 / 14))

    # ---- f: population-average PETHs ----
    panel_letter(fig, 0.0, 0.355, 'f')
    ax = add_ax(fig, 0.06, 0.395, 0.37, 0.065)
    for peth, col in [(stint, STINT_COL), (seq, SEQ_COL)]:
        ax.plot(peth.mean(0), color=col, lw=1)
    for b in np.cumsum(list(seg_lens.values())):
        ax.axvline(b, ls='--', lw=1, color='grey')
    ax.set_xlim(0, seq.shape[1])
    ax.set_xticks([])
    ax.set_ylim(*peth_ylim(ax))
    ax.set_ylabel('avg. z-scr rates')

    ax = add_ax(fig, 0.47, 0.367, 0.28, 0.095)
    for peth, col, name, ytxt in [(stint, STINT_COL, 'stim/integ', 0.42),
                                  (seq, SEQ_COL, 'sequence', 0.07)]:
        trace = condition_average(peth, seg_lens).mean(0)
        ax.plot(trace, color=col, lw=1)
        ax.text(0.45 * len(trace), ytxt, name, color=col, fontsize=11)
    ax.axvline(len(trace) - 1, ls='--', lw=1, color='grey')
    ax.set_xlim(0, len(trace) - 1)
    ax.set_xticks([])
    ax.set_ylim(*peth_ylim(ax))
    ax.set_ylabel('avg (trial-types)')

    # ---- g: average firing rate ----
    if composition:
        panel_letter(fig, 0.78, 0.345, 'g')
        ax = add_ax(fig, 0.84, 0.355, 0.14, 0.09)
        ax.bar(GROUP_BAR_LABELS, avg_fr,
               color=[f'C{i}' for i in range(len(avg_fr))])
        ax.set_ylabel('average firing rate')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # ---- h: sequence neurons across trial types ----
    panel_letter(fig, 0.0, 0.515, 'h')
    fig.text(0.27, 0.48, 'sequence neurons', color=SEQ_COL, fontsize=12,
             ha='center', va='top')
    fig.text(0.02, 1 - 0.625, 'correln cross trial-type', rotation=90, fontsize=11,
             ha='center', va='center')
    ax = add_ax(fig, 0.085, 0.545, 0.19, 0.19 * 12 / 14)
    im = concat_corr(ax, cc_seq, seg_lens, labels, xlabels=False)
    ax.set_title('time', fontsize=7, loc='right', pad=2)
    cax = add_ax(fig, 0.29, 0.545, 0.008, 0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Pearson r', fontsize=7)

    ax = add_ax(fig, 0.34, 0.61, 0.11, 0.11 * 12 / 14)
    mat = overall_seq.copy()
    np.fill_diagonal(mat, 0)
    im = ax.imshow(mat, cmap='Reds')
    names_c = [labels[k] for k in CONDITIONS]
    ax.set_xticks(range(len(CONDITIONS)), names_c, rotation=60, ha='right',
                  fontsize=6.5)
    ax.set_yticks(range(len(CONDITIONS)), names_c, fontsize=6.5)
    cax = add_ax(fig, 0.40, 0.595, 0.05, 0.006)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.locator = mpl.ticker.MaxNLocator(3)
    cb.update_ticks()
    cax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_title('Pearson r', fontsize=7, pad=10)

    fig.text(0.02, 1 - 0.8, 'ctxt seq similarity', rotation=90, fontsize=11,
             ha='center', va='center')
    print('2D cross-correlograms ...')
    cross_corr_grid(fig, (0.085, 0.725, 0.17, 0.17 * 12 / 14), seq, seg_lens,
                    CONDITIONS, labels)

    # ---- i: shuffles ----
    panel_letter(fig, 0.48, 0.515, 'i')
    fig.text(0.61, 0.48, 'sequence neuron shuffles', color=SHUF_COL,
             fontsize=11, ha='center', va='top')
    ax = add_ax(fig, 0.515, 0.545, 0.185, 0.185 * 12 / 14)
    concat_corr(ax, cc_shuf, seg_lens, labels, ylabels=False)

    # ---- j: stim/integ ----
    panel_letter(fig, 0.745, 0.515, 'j')
    fig.text(0.88, 0.48, 'stimulus/integrator neurons', color=STINT_COL,
             fontsize=11, ha='center', va='top')
    ax = add_ax(fig, 0.785, 0.545, 0.185, 0.185 * 12 / 14)
    concat_corr(ax, cc_stint, seg_lens, labels, ylabels=False)

    # ---- k: regional composition ----
    panel_letter(fig, 0.29, 0.78, 'k')
    for x0, kind, title, col in [
            (0.33, 'seq', 'sequence', SEQ_COL),
            (0.55, 'stint', 'stim/integ', STINT_COL)]:
        ax_pie = add_ax(fig, x0, 0.83, 0.155, 0.155 * 12 / 14, projection='polar')
        if f'{kind}_mask' in d:
            region_pie(ax_pie, r['Beryl'], r['cols'], d[f'{kind}_mask'])
        else:
            clusters = SEQ_CLUSTERS if kind == 'seq' else STINT_CLUSTERS
            ax_dummy = add_ax(fig, x0, 0.83, 0.01, 0.01)
            plot_rm_cluster_profile(clusters, canonical_order=True, nclus=100,
                                    top_wedge_labels=15, axs=(ax_pie, ax_dummy))
            ax_dummy.remove()
        ax_pie.set_title(title, color=col, fontsize=11, pad=32)

    # ---- l: cell composition ----
    if composition:
        panel_letter(fig, 0.715, 0.765, 'l')
        ax = add_ax(fig, 0.745, 0.79, 0.21, 0.21 * 12 / 14)
        # first wedge (sequences) ends at 12 o'clock, groups follow clockwise
        start = 90 + 360 * n_cells[0] / n_cells.sum()
        wedges, texts, autotexts = ax.pie(
            n_cells, labels=names, autopct='%.1f%%', startangle=start,
            counterclock=False, textprops=dict(fontsize=10))
        for t, w in zip(texts, wedges):
            t.set_color(w.get_facecolor())
        for t in autotexts:
            t.set_fontsize(7)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'png']:
            fig.savefig(out_dir / f'fig5.{ext}', dpi=300)
        save_panels(fig, out_dir)
        print(f'saved fig5 and panels to {out_dir}')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['old', 'new'], default='old',
                        help="'old': cached regional_group stack; "
                             "'new': concat_cvTrue.npy in the repo")
    args = parser.parse_args()

    figs_dir = Path(pth_dmn.parent, 'figs')
    if args.data == 'old':
        make_figure(load_data(), figs_dir / 'figure5')
    else:
        out_dir = figs_dir / 'figure5_new'
        d = load_new_data()
        plot_cluster_id(d['raw'], out_dir / 'cluster_id')
        print(f"{len(d['seq_peth'])} sequence cells, "
              f"{len(d['stint_peth'])} stim/integ cells")
        make_figure(d, out_dir, composition=False)
