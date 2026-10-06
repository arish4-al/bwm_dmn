from dmn_ari import sum_for_key
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path    
from scipy.stats import pearsonr
from dmn_bwm import pth_dmn
from matplotlib.gridspec import GridSpec


def variance_by_clusters(r, clusters):
    """
    r['concat_z']: (cells × time)
    r['acs']: cluster id per cell

    clusters: list of cluster IDs to include

    Returns
    -------
    total_var : float
        Total variance of full matrix (using global mean)
    sub_var : float
        Total variance of subset (using same global mean)
    n_cells_subset : int
        Number of cells in subset
    """

    X = np.asarray(r['concat_z'])
    clus_ids = np.asarray(r['acs']).astype(int)

    clusters = np.array(clusters, dtype=int)
    mask = np.isin(clus_ids, clusters)

    if not mask.any():
        raise ValueError(f"No cells found for clusters {clusters.tolist()}")

    X_sub = X[mask]
    mu = X.mean()

    # total_var = np.sum((X - mu) ** 2)
    sub_var = np.sum((X_sub - mu) ** 2)

    n_cells_subset = mask.sum()

    return sub_var, n_cells_subset


def firing_rate_by_clusters(r, clusters, rastertype, ylim=None):
    """
    Total (summed) firing rate per group and average firing rate per neuron in each group,
    and plot the average PETH across all included cells as a function of time.

    r['concat']: (cells × time)
    r['acs']: cluster id per cell
    clusters: list of cluster IDs to include

    Returns
    -------
    total_fr_timeseries : ndarray, shape (n_time,)
        At each time point, sum of firing rates across all neurons in the group
        (population total firing rate per time bin).
    avg_fr_per_neuron : float
        Mean firing rate per neuron in the group (mean over time and over neurons).
    """

    X = np.asarray(r['concat'])
    clus_ids = np.asarray(r['acs']).astype(int)

    clusters = np.array(clusters, dtype=int)
    mask = np.isin(clus_ids, clusters)

    if not mask.any():
        raise ValueError(f"No cells found for clusters {clusters.tolist()}")

    X_sub = X[mask]  # (n_cells_subset, n_time)

    # Total firing rate per time: sum across neurons at each time
    total_fr_timeseries = np.mean(np.sum(X_sub, axis=0))

    # Average firing rate per neuron (mean over all cells and time in group)
    avg_fr_per_neuron = np.mean(X_sub)

    # Average PETH across all cells as a function of time
    avg_peth = np.mean(X_sub, axis=0)

    plt.figure()
    plt.plot(avg_peth)
    plt.xlabel('Time (bins)')
    plt.ylabel('Mean firing rate')
    plt.title(f'Average PETH {rastertype} cells')
    plt.savefig(Path(pth_dmn.parent, 'figs', f'avg_peth_{rastertype}.svg'), dpi=150)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()

    return total_fr_timeseries, avg_fr_per_neuron


def overall_corr_matrix(peth, data_lengths, conditions,
                        rng=None, return_p=False, n_perm=1000):

    for key in conditions:
        if key not in data_lengths:
            raise KeyError(f"{key} not found in r['len']")

    if rng is None:
        rng = np.random.default_rng(0)

    segments = {}
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")

        segments[key] = peth[:, start:end]

    n = len(conditions)
    corr_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)

    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):

            seg_i = segments[ci]
            seg_j = segments[cj]

            xi = seg_i.ravel()
            yj = seg_j.ravel()

            if (not np.isfinite(xi).all() or
                not np.isfinite(yj).all() or
                np.std(xi) == 0 or
                np.std(yj) == 0):
                continue

            r_obs, _ = pearsonr(xi, yj)
            corr_mat[i, j] = r_obs

            if return_p:

                xi_z = (xi - xi.mean()) / xi.std()
                yj_z = (seg_j - yj.mean()) / yj.std()

                N = xi_z.size
                exceed = 0

                for _ in range(n_perm):
                    perm = rng.permutation(seg_j.shape[0])
                    y_perm = yj_z[perm].ravel()
                    r_null = (y_perm @ xi_z) / (N - 1)
                    if abs(r_null) >= abs(r_obs):
                        exceed += 1

                p_mat[i, j] = (1 + exceed) / (n_perm + 1)

    return (corr_mat, p_mat) if return_p else corr_mat


def _annotate_pvals(ax, p_mat):
    for i in range(p_mat.shape[0]):
        for j in range(p_mat.shape[1]):
            p = p_mat[i, j]
            if np.isfinite(p):
                ax.text(j, i, f"{p:.2f}", ha='center', va='center', fontsize=7, color='black')


from matplotlib.gridspec import GridSpec

def plot_corr_mats(
    corr_mat, conditions, suptitle,
    p_mat=None, annotate_pvals=False, save_name=None
):
    fig = plt.figure(figsize=(4, 4))

    gs = GridSpec(
        1, 2,
        width_ratios=[1, 0.05],
        wspace=0.1
    )

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    mat = corr_mat.copy()
    np.fill_diagonal(mat, 0)

    im = ax.imshow(mat, cmap='Reds')
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)

    if annotate_pvals and p_mat is not None:
        p_display = p_mat.copy()
        np.fill_diagonal(p_display, np.nan)
        _annotate_pvals(ax, p_display)

    fig.colorbar(im, cax=cax, label='overall Pearson r')

    if suptitle is not None:
        fig.suptitle(suptitle, y=1.05)

    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name),
                    dpi=200, bbox_inches='tight')

    plt.show()


def bin_autocorr_mats(
    peth,
    data_lengths,
    conditions,
    shuffle_columns=False,
):
    """
    Per-condition bin×bin correlation matrices (corrcoef across cells, rowvar=False).

    Pass ``peth`` and ``data_lengths`` from ``plot_raster_subset`` with
    ``return_processed=True`` (e.g. ``out['peth']``, ``out['seg_lens']``), with
    ``conditions`` matching ``segments_subset``. Row order and shuffles are fixed in
    that plot call.

    If shuffle_columns is True, time columns are permuted globally on ``peth`` (null).
    """
    if shuffle_columns:
        peth = np.asarray(peth, dtype=float).copy()
        perm = np.random.permutation(peth.shape[1])
        peth = peth[:, perm]
    else:
        peth = np.asarray(peth, dtype=float)

    mats = {}
    cond_blocks = []

    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")

        seg = peth[:, start:end]
        cond_blocks.append(seg)

        corr = np.corrcoef(seg, rowvar=False)

        bad = (~np.isfinite(seg).all(axis=0)) | (np.std(seg, axis=0) == 0)
        if np.any(bad):
            corr[bad, :] = np.nan
            corr[:, bad] = np.nan

        mats[key] = corr

    # ==========================================
    # NEW: condition-averaged autocorrelation
    # ==========================================
    if len(cond_blocks) > 0:
        cond_blocks = np.stack(cond_blocks, axis=0)      # (n_cond × cells × bins)
        avg_mat = cond_blocks.mean(axis=0)               # (cells × bins)

        avg_corr = np.corrcoef(avg_mat, rowvar=False)

        bad = (~np.isfinite(avg_mat).all(axis=0)) | (np.std(avg_mat, axis=0) == 0)
        if np.any(bad):
            avg_corr[bad, :] = np.nan
            avg_corr[:, bad] = np.nan

        mats["avg_conditions"] = avg_corr

    return mats


def plot_bin_corr_grid(
    mats,
    conditions,
    title,
    save_name=None,
    ncols=3,
    cmap='coolwarm'
):
    n = len(conditions)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2 * ncols, 2 * nrows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    im = None
    for idx, key in enumerate(conditions):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        mat = mats[key].copy()
        np.fill_diagonal(mat, 0)

        im = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, nrows * ncols):
        axes.flat[idx].axis('off')

    if im is not None:
        fig.colorbar(im, ax=axes, label='Pearson r', shrink=0.85, pad=0.02)

    fig.suptitle(title)

    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)

    plt.show()

    # ==========================================
    # NEW: separate plot for condition-average
    # ==========================================
    if "avg_conditions" in mats:
        avg_corr = mats["avg_conditions"].copy()
        np.fill_diagonal(avg_corr, 0)

        fig2, ax2 = plt.subplots(figsize=(3, 3))
        im2 = ax2.imshow(avg_corr, vmin=-1, vmax=1, cmap=cmap)

        ax2.set_title("avg_conditions")
        ax2.set_xticks([])
        ax2.set_yticks([])

        fig2.colorbar(im2, ax=ax2, label='Pearson r', shrink=0.8)

        fig2.tight_layout()

        if save_name is not None:
            base = Path(save_name).stem
            fig2.savefig(
                Path(pth_dmn.parent, 'figs', f"{base}_avg_conditions.svg"),
                dpi=200
            )

        plt.show()


def autocorrelograms(peth, data_lengths, conditions, max_lag=None):
    acorrs = {}
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")
        seg = peth[:, start:end]
        mean_trace = np.nanmean(seg, axis=0)
        mean_trace = mean_trace - np.nanmean(mean_trace)
        n = mean_trace.size
        max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
        full = np.correlate(mean_trace, mean_trace, mode='full')
        mid = n - 1
        denom = full[mid]
        if denom == 0 or not np.isfinite(denom):
            ac = np.full(max_lag + 1, np.nan)
        else:
            ac = full[mid:mid + max_lag + 1] / denom
        acorrs[key] = ac
    return acorrs


def plot_autocorr_grid(acorrs, conditions, title, save_name=None, ncols=3):
    n = len(conditions)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, key in enumerate(conditions):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ac = acorrs[key]
        ax.plot(ac, color='black')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(key)
        ax.set_xlabel('lag (bins)')
        ax.set_ylabel('corr')

    for idx in range(n, nrows * ncols):
        axes.flat[idx].axis('off')

    fig.suptitle(title)
    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)
    plt.show()


def crosscorrelograms(
    peth,
    data_lengths,
    conditions,
    max_lag=None,
    shuffle_time=False,
    rng=None,
    normalize=True,
    normalize_mode='per_condition',
):
    if rng is None:
        rng = np.random.default_rng(0)

    traces = {}
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")
        seg = peth[:, start:end]
        if shuffle_time:
            seg = np.vstack([row[rng.permutation(row.size)] for row in seg])
        trace = np.nanmean(seg, axis=0)
        trace = trace - np.nanmean(trace)
        traces[key] = trace

    global_denom = None
    if normalize_mode == 'global' and normalize == True:
        concat = np.concatenate([traces[k] for k in conditions])
        global_denom = np.sum(concat ** 2)

    cross = {}
    for a in conditions:
        for b in conditions:
            xa = traces[a]
            xb = traces[b]
            n = min(xa.size, xb.size)
            xa = xa[:n]
            xb = xb[:n]
            max_lag_use = n - 1 if max_lag is None else min(max_lag, n - 1)
            full = np.correlate(xa, xb, mode='full')
            mid = n - 1
            if normalize_mode == 'global' and global_denom is not None:
                denom = global_denom
            elif normalize == True:
                denom = np.sqrt(np.sum(xa ** 2) * np.sum(xb ** 2))
            else:
                denom = 0
            if denom == 0 or not np.isfinite(denom):
                # cc = np.full(2 * max_lag_use + 1, np.nan)
                cc = full[mid - max_lag_use:mid + max_lag_use + 1]
            else:
                cc = full[mid - max_lag_use:mid + max_lag_use + 1] / denom
            cross[(a, b)] = cc
    return cross


def plot_crosscorr_grid_1d(corrs, conditions, title, save_name=None):
    n = len(conditions)
    fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n), sharex=True, sharey=True, constrained_layout=True)

    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            ax = axes[i, j]
            cc = corrs.get((a, b))
            if cc is None:
                ax.axis('off')
                continue
            ax.plot(cc, color='black', linewidth=0.8)
            ax.axhline(0, color='gray', linewidth=0.5)
            if i == 0:
                ax.set_title(b)
            if j == 0:
                ax.set_ylabel(a)
            ax.tick_params(labelbottom=False, labelleft=False)

    fig.supxlabel('lag (bins)')
    fig.supylabel('corr')
    fig.suptitle(title)
    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)
    plt.show()


def peak_matrix(corrs, conditions, use_abs=True):
    n = len(conditions)
    mat = np.full((n, n), np.nan)
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            cc = corrs.get((a, b))
            if cc is None:
                continue
            vals = np.abs(cc) if use_abs else cc
            mat[i, j] = np.nanmax(vals)
    return mat


def plot_peak_matrix(mat, conditions, title, save_name=None, vmin=0, vmax=1, cmap='Reds'):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions)
    fig.colorbar(im, ax=ax, label='peak corr', fraction=0.046, pad=0.04)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)
    plt.show()
    

def _corr2d(a, b, use_fft=True, smooth_sigma=None, mask_center=False):
    from scipy.signal import correlate2d, fftconvolve
    from scipy.ndimage import gaussian_filter

    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None

    if smooth_sigma is not None:
        a = gaussian_filter(a, smooth_sigma, mode='nearest')
        b = gaussian_filter(b, smooth_sigma, mode='nearest')

    if use_fft:
        corr = fftconvolve(a, b[::-1, ::-1], mode='full')
    else:
        corr = correlate2d(a, b, mode='full')

    # --- OPTION 1: remove zero-lag dominance ---
    if mask_center:
        ci = corr.shape[0] // 2
        cj = corr.shape[1] // 2
        corr[ci, cj] = 0.0   # or np.nan if you prefer

    return corr
    

def image_autocorrs(
    peth,
    data_lengths,
    conditions,
    use_fft=True,
    smooth_sigma=None,
    normalize=True,
    normalize_mode='per_condition',
):
    global_center = None
    if normalize and normalize_mode == 'global':
        segments = []
        for key in conditions:
            start = sum_for_key(data_lengths, key)
            end = sum_for_key(data_lengths, key, after=True)
            if isinstance(start, str) or isinstance(end, str):
                raise KeyError(f"Invalid key for r['len']: {key}")
            segments.append(peth[:, start:end])
        concat_img = np.concatenate(segments, axis=1)
        global_corr = _corr2d(concat_img, concat_img, use_fft=use_fft, smooth_sigma=smooth_sigma)
        if global_corr is not None:
            global_center = global_corr[global_corr.shape[0] // 2, global_corr.shape[1] // 2]

    acorrs = {}
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")
        img = peth[:, start:end]
        corr = _corr2d(img, img, use_fft=use_fft, smooth_sigma=smooth_sigma)
        if corr is None:
            acorrs[key] = None
            continue
        if normalize:
            center = (
                global_center
                if normalize_mode == 'global' and global_center is not None
                else corr[corr.shape[0] // 2, corr.shape[1] // 2]
            )
            acorrs[key] = corr / center if center != 0 else corr
        else:
            acorrs[key] = corr
    return acorrs


def image_crosscorrs(
    peth,
    data_lengths,
    conditions,
    use_fft=True,
    smooth_sigma=None,
    shuffle_rows=False,
    rng=None,
    normalize=True,
    normalize_mode='per_condition',
):
    if rng is None:
        rng = np.random.default_rng(0)

    mats = {}
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")
        seg = peth[:, start:end]
        if shuffle_rows:
            seg = seg[rng.permutation(seg.shape[0])]
        mats[key] = seg

    global_center = None
    if normalize and normalize_mode == 'global':
        concat_img = np.concatenate([mats[k] for k in conditions], axis=1)
        global_corr = _corr2d(concat_img, concat_img, use_fft=use_fft, smooth_sigma=smooth_sigma)
        if global_corr is not None:
            global_center = global_corr[global_corr.shape[0] // 2, global_corr.shape[1] // 2]

    cross = {}
    for a in conditions:
        for b in conditions:
            corr = _corr2d(mats[a], mats[b], use_fft=use_fft, smooth_sigma=smooth_sigma)
            if corr is None:
                cross[(a, b)] = None
                continue
            if normalize:
                center = (
                    global_center
                    if normalize_mode == 'global' and global_center is not None
                    else corr[corr.shape[0] // 2, corr.shape[1] // 2]
                )
                cross[(a, b)] = corr / center if center != 0 else corr
            else:
                cross[(a, b)] = corr
    return cross


def plot_image_corr_grid(corrs, conditions, title, save_name=None, ncols=3, cmap='coolwarm', vmin=-1, vmax=1):
    n = len(conditions)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    im = None
    for idx, key in enumerate(conditions):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        corr = corrs.get(key)
        if corr is None:
            ax.axis('off')
            continue
        im = ax.imshow(
            corr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            interpolation='nearest',
        )
        ax.set_title(key)
        ax.set_xlabel('bin lag')
        ax.set_ylabel('cell lag')

    for idx in range(n, nrows * ncols):
        axes.flat[idx].axis('off')

    if im is not None:
        fig.colorbar(im, ax=axes, label='2D corr', shrink=0.85, pad=0.02)
    fig.suptitle(title)
    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)
    plt.show()


def plot_crosscorr_matrix_grid(
    crosscorrs,
    conditions,
    title,
    save_name=None,
    cmap='coolwarm',
    vmin=None,
    vmax=None,
    log_scale=False,     # NEW
    linthresh=1e-2,      # NEW (for symmetric log)
):
    n = len(conditions)
    fig, axes = plt.subplots(
        n, n,
        figsize=(1 * n, 1 * n),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # --- choose normalization ---
    if log_scale:
        norm = matplotlib.colors.SymLogNorm(
            linthresh=linthresh,
            vmin=vmin,
            vmax=vmax,
            base=10,
        )
    else:
        norm = None

    im = None
    for i, a in enumerate(conditions):
        for j, b in enumerate(conditions):
            ax = axes[i, j]
            corr = crosscorrs.get((a, b))
            if corr is None:
                ax.axis('off')
                continue

            im = ax.imshow(
                corr,
                cmap=cmap,
                vmin=None if log_scale else vmin,
                vmax=None if log_scale else vmax,
                norm=norm,
                aspect='auto',
                interpolation='nearest',
            )

            # column titles
            if i == 0:
                ax.set_title(
                    b,
                    fontsize=12,
                    rotation=25,
                    ha='center',
                    va='bottom',
                )

            # row labels
            if j == 0:
                ax.set_ylabel(
                    a,
                    fontsize=12,
                    rotation=0,
                    ha='right',
                    va='center',
                )

            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.supxlabel('time lag')
    fig.supylabel('cell lag')

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)

    fig.suptitle(title)

    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)

    plt.show()


def concat_bin_autocorr(
    peth,
    data_lengths,
    conditions,
    shuffle_columns=False,
    shuffle_columns_within_condition=False,
):
    """
    Bin×bin correlation after concatenating condition segments along time.

    Same inputs as ``bin_autocorr_mats``: ``out['peth']`` and ``out['seg_lens']`` from
    ``plot_raster_subset(..., return_processed=True)``.

    shuffle_columns: global column permutation (null). shuffle_columns_within_condition:
    permute time bins independently within each segment after extraction.
    """
    if shuffle_columns or shuffle_columns_within_condition:
        peth = np.asarray(peth, dtype=float).copy()
    else:
        peth = np.asarray(peth, dtype=float)

    if shuffle_columns:
        perm = np.random.permutation(peth.shape[1])
        peth = peth[:, perm]

    segments = []
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")

        seg = peth[:, start:end]
        if shuffle_columns_within_condition:
            T = seg.shape[1]
            if T > 0:
                perm = np.random.permutation(T)
                seg = seg[:, perm].copy()
        segments.append(seg)

    concat_seg = np.concatenate(segments, axis=1)
    corr = np.corrcoef(concat_seg, rowvar=False)

    bad = (~np.isfinite(concat_seg).all(axis=0)) | (np.std(concat_seg, axis=0) == 0)
    if np.any(bad):
        corr[bad, :] = np.nan
        corr[:, bad] = np.nan

    return corr


def plot_concat_bin_corr(corr_mat, conditions, data_lengths, title, save_name=None, cmap='coolwarm',
                         log_intensity=False, log_linthresh=0.05, log_base=10.0):

    fig, ax = plt.subplots(figsize=(6, 5))
    np.fill_diagonal(corr_mat, 0)
    
    if log_intensity:
        import matplotlib
        norm = matplotlib.colors.SymLogNorm(
            linthresh=log_linthresh, vmin=-1, vmax=1, base=log_base
        )

        im = ax.imshow(corr_mat, cmap=cmap, norm=norm)
        cbar_label = 'Pearson r (symlog)'
    else:
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap=cmap)
        cbar_label = 'Pearson r'

    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    # im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap=cmap)

    boundaries = [0]
    centers = []
    current = 0
    for key in conditions:
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        seg_len = end - start
        current += seg_len
        boundaries.append(current)
        centers.append(current - seg_len / 2)

    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color='k', linewidth=0.5, alpha=0.4)
        ax.axhline(b - 0.5, color='k', linewidth=0.5, alpha=0.4)

    ax.set_xticks(centers)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.set_yticks(centers)
    ax.set_yticklabels(conditions)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if save_name is not None:
        fig.savefig(Path(pth_dmn.parent, 'figs', save_name), dpi=200)
    plt.show()


def per_cell_condition_corr(peth, data_lengths, conditions):
    if len(conditions) != 2:
        raise ValueError("conditions must have exactly two entries")
    for key in conditions:
        if key not in data_lengths:
            raise KeyError(f"{key} not found in r['len']")

    def _segment_for_key(data_lengths, key):
        start = sum_for_key(data_lengths, key)
        end = sum_for_key(data_lengths, key, after=True)
        if isinstance(start, str) or isinstance(end, str):
            raise KeyError(f"Invalid key for r['len']: {key}")
        return start, end

    start0, end0 = _segment_for_key(data_lengths, conditions[0])
    start1, end1 = _segment_for_key(data_lengths, conditions[1])

    x = peth[:, start0:end0]
    y = peth[:, start1:end1]

    r_vals = []
    p_vals = []
    for xi, yi in zip(x, y):
        if (not np.isfinite(xi).all() or not np.isfinite(yi).all() or
                np.std(xi) == 0 or np.std(yi) == 0):
            r_vals.append(np.nan)
            p_vals.append(np.nan)
            continue
        r, p = pearsonr(xi, yi)
        r_vals.append(r)
        p_vals.append(p)

    x_all = x.ravel()
    y_all = y.ravel()
    if (np.isfinite(x_all).all() and np.isfinite(y_all).all() and
            np.std(x_all) != 0 and np.std(y_all) != 0):
        overall_r, overall_p = pearsonr(x_all, y_all)
    else:
        overall_r, overall_p = np.nan, np.nan

    return np.array(r_vals), np.array(p_vals), overall_r, overall_p


def decode_trial_conditions_over_time(
    trial_types,
    seq_peth,
    condition_bin_lengths,
    seq_peth_test=None,
    decode_mode='multiclass',
    mistake_condition_names=None,
    classifier='logistic',
    class_weight=None,
    balance_train_classes=False,
    svm_C=1.0,
    svm_gamma='scale',
    mlp_hidden_layer_sizes=(32,),
    mlp_alpha=1e-4,
    mlp_max_iter=500,
    random_state=0,
):
    """
    Decode trial condition from sequence-cell population activity at each time bin.

    Parameters
    ----------
    trial_types : sequence of str
        Ordered trial-condition names (e.g. ``conditions_all``).
    seq_peth : array-like, shape (n_cells, n_time_bins_total)
        Concatenated sequence-cell activity matrix used as training data.
    condition_bin_lengths : dict[str, int] | sequence[int]
        Number of bins for each condition in ``trial_types``.
        - If dict/OrderedDict: treated as the full concatenation layout (e.g.
          ``r['len']``), and each condition segment is located by absolute
          start/end position in ``seq_peth``.
        - If sequence: treated as lengths in ``trial_types`` order, assuming
          ``seq_peth`` is already restricted to only those conditions in order.
        This argument is required; no inference from ``seq_peth`` is performed.
    seq_peth_test : array-like, shape (n_cells, n_time_bins_total), optional
        Independent test split (e.g. test-half average). If provided, decoding
        uses ``seq_peth`` for training and ``seq_peth_test`` for testing at each
        time bin. If None, training/testing are both done on ``seq_peth`` using
        leave-one-time-bin-out within each condition.
    decode_mode : {'multiclass', 'mistake_vs_non_mistake'}, default='multiclass'
        Labeling scheme for decoding.
    mistake_condition_names : sequence[str] | None, optional
        Used only when ``decode_mode='mistake_vs_non_mistake'``. If provided,
        these trial-type names are assigned to the ``mistake`` class; all other
        trial types are assigned to ``non_mistake``. If None, mistake classes are
        inferred as names containing the substring ``'mistake'`` (case-insensitive).
    classifier : {'logistic', 'nearest_centroid', 'svm_linear', 'svm_rbf', 'mlp'}, default='logistic'
        Decoder used per time-bin fold.
    class_weight : dict | 'balanced' | None, default=None
        Class weights passed to ``sklearn.linear_model.LogisticRegression`` when
        ``classifier='logistic'`` and to SVM classifiers
        (``svm_linear``/``svm_rbf``). Useful for imbalance in
        ``mistake_vs_non_mistake`` mode.
    balance_train_classes : bool, default=False
        If True, training samples are balanced by random undersampling so each
        decoded class contributes the same number of samples per fold. This is
        most useful in ``mistake_vs_non_mistake`` mode.
    svm_C : float, default=1.0
        Regularization strength for SVM classifiers.
    svm_gamma : {'scale', 'auto'} | float, default='scale'
        Kernel coefficient for ``svm_rbf``.
    mlp_hidden_layer_sizes : tuple[int, ...], default=(32,)
        Hidden-layer sizes for ``mlp`` decoder.
    mlp_alpha : float, default=1e-4
        L2 penalty for ``mlp`` decoder.
    mlp_max_iter : int, default=500
        Max iterations for ``mlp`` decoder.
    random_state : int, default=0
        Random seed for deterministic logistic-regression behavior.

    Returns
    -------
    results : dict
        Keys:
        - ``trial_types``: condition names
        - ``class_names``: decoded class names
        - ``condition_to_class``: per-condition class assignment
        - ``decode_mode``: labeling scheme used
        - ``usable_bins``: number of aligned bins used per condition
        - ``bin_edges``: cumulative bin edges in the original concatenated matrix
        - ``accuracy_by_time``: array, shape (usable_bins,)
        - ``balanced_accuracy_by_time``: array, shape (usable_bins,)
        - ``per_class_recall_by_time``: array, shape (usable_bins, n_decoded_classes)
        - ``train_class_counts_by_time``: array, shape (usable_bins, n_decoded_classes)
        - ``chance_level``: float (1 / n_decoded_classes)
        - ``y_true_by_time``: array, shape (usable_bins, n_conditions)
        - ``y_pred_by_time``: array, shape (usable_bins, n_conditions)
        - ``confusion_by_time``: array, shape (usable_bins, n_decoded_classes, n_decoded_classes)
          Rows are true labels, columns are predicted labels.
    """
    trial_types = list(trial_types)
    if len(trial_types) < 2:
        raise ValueError("trial_types must contain at least 2 conditions")

    X_train_full = np.asarray(seq_peth, dtype=float)
    if X_train_full.ndim != 2:
        raise ValueError("seq_peth must be 2D with shape (cells, time_bins)")

    n_cells, n_total_bins = X_train_full.shape
    if n_cells < 1 or n_total_bins < 2:
        raise ValueError("seq_peth must have at least 1 cell and 2 time bins")

    if seq_peth_test is not None:
        X_test_full = np.asarray(seq_peth_test, dtype=float)
        if X_test_full.ndim != 2:
            raise ValueError("seq_peth_test must be 2D with shape (cells, time_bins)")
        if X_test_full.shape[0] != n_cells:
            raise ValueError(
                "seq_peth and seq_peth_test must have the same number of cells"
            )
    else:
        X_test_full = None

    n_conditions = len(trial_types)
    if decode_mode not in ('multiclass', 'mistake_vs_non_mistake'):
        raise ValueError("decode_mode must be 'multiclass' or 'mistake_vs_non_mistake'")

    if decode_mode == 'multiclass':
        cond_decode_labels = np.arange(n_conditions, dtype=int)
        class_names = list(trial_types)
    else:
        if mistake_condition_names is None:
            mistake_set = {k for k in trial_types if 'mistake' in k.lower()}
        else:
            mistake_set = set(mistake_condition_names)
        cond_decode_labels = np.array(
            [1 if k in mistake_set else 0 for k in trial_types],
            dtype=int
        )
        class_names = ['non_mistake', 'mistake']
        if np.unique(cond_decode_labels).size < 2:
            raise ValueError(
                "mistake_vs_non_mistake requires at least one mistake and one non-mistake "
                "trial type in trial_types"
            )
    n_decode_classes = len(class_names)

    def _segments_from_layout(X_full):
        n_total_bins_local = X_full.shape[1]
        lengths_local = None
        if isinstance(condition_bin_lengths, dict):
            missing = [k for k in trial_types if k not in condition_bin_lengths]
            if missing:
                raise KeyError(f"Missing condition_bin_lengths for: {missing}")
            edge_pairs = []
            for key in trial_types:
                start = sum_for_key(condition_bin_lengths, key)
                end = sum_for_key(condition_bin_lengths, key, after=True)
                if isinstance(start, str) or isinstance(end, str):
                    raise KeyError(f"Invalid key in condition_bin_lengths: {key}")
                if start < 0 or end > n_total_bins_local:
                    raise ValueError(
                        f"Condition {key} slice [{start}, {end}) exceeds "
                        f"time axis length {n_total_bins_local}"
                    )
                edge_pairs.append((int(start), int(end)))
            lengths_local = [end - start for start, end in edge_pairs]
            segments_local = [X_full[:, start:end] for start, end in edge_pairs]
        else:
            lengths_local = [int(v) for v in condition_bin_lengths]
            if len(lengths_local) != n_conditions:
                raise ValueError(
                    "condition_bin_lengths length must match len(trial_types)"
                )
            if any(v <= 0 for v in lengths_local):
                raise ValueError("All condition lengths must be positive integers")
            if sum(lengths_local) != n_total_bins_local:
                raise ValueError(
                    f"sum(condition_bin_lengths)={sum(lengths_local)} does not match "
                    f"time axis length {n_total_bins_local}"
                )
            edges_local = np.concatenate([[0], np.cumsum(lengths_local)])
            segments_local = [
                X_full[:, edges_local[i]:edges_local[i + 1]]
                for i in range(n_conditions)
            ]
        return segments_local, lengths_local

    condition_segments_train, lengths = _segments_from_layout(X_train_full)
    if X_test_full is not None:
        condition_segments_test, _ = _segments_from_layout(X_test_full)
    else:
        condition_segments_test = condition_segments_train

    # Align to the shortest condition so each moment is comparable across conditions.
    usable_bins = int(
        min(
            min(seg.shape[1] for seg in condition_segments_train),
            min(seg.shape[1] for seg in condition_segments_test),
        )
    )
    if usable_bins < 2:
        raise ValueError("Need at least 2 aligned bins per condition to decode over time")
    condition_segments_train = [seg[:, :usable_bins] for seg in condition_segments_train]
    condition_segments_test = [seg[:, :usable_bins] for seg in condition_segments_test]

    y_true_by_time = []
    y_pred_by_time = []
    confusion_by_time = np.zeros((usable_bins, n_decode_classes, n_decode_classes), dtype=int)
    accuracy_by_time = np.full(usable_bins, np.nan)
    per_class_recall_by_time = np.full((usable_bins, n_decode_classes), np.nan)
    balanced_accuracy_by_time = np.full(usable_bins, np.nan)
    train_class_counts_by_time = np.zeros((usable_bins, n_decode_classes), dtype=int)
    rng_balance = np.random.default_rng(random_state)

    for t in range(usable_bins):
        X_test = np.stack([seg[:, t] for seg in condition_segments_test], axis=0)
        y_test = cond_decode_labels.copy()

        train_vectors = []
        train_labels = []
        for cond_idx, seg in enumerate(condition_segments_train):
            if X_test_full is None:
                # Same-matrix decoding: leave held-out bin out of train set.
                xtr = np.delete(seg, t, axis=1).T
            else:
                # Independent split decoding: keep all train bins.
                xtr = seg.T
            train_vectors.append(xtr)
            train_labels.append(np.full(xtr.shape[0], cond_decode_labels[cond_idx], dtype=int))

        X_train = np.vstack(train_vectors)
        y_train = np.concatenate(train_labels)

        if balance_train_classes:
            present = np.unique(y_train)
            cls_counts = [np.sum(y_train == cls) for cls in present]
            if len(cls_counts) >= 2 and min(cls_counts) > 0:
                n_keep = int(min(cls_counts))
                keep_idx = []
                for cls in present:
                    idx_cls = np.flatnonzero(y_train == cls)
                    if idx_cls.size > n_keep:
                        idx_cls = rng_balance.choice(idx_cls, size=n_keep, replace=False)
                    keep_idx.append(idx_cls)
                keep_idx = np.concatenate(keep_idx)
                rng_balance.shuffle(keep_idx)
                X_train = X_train[keep_idx]
                y_train = y_train[keep_idx]

        for cls in range(n_decode_classes):
            train_class_counts_by_time[t, cls] = int(np.sum(y_train == cls))

        mu = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        X_train_z = (X_train - mu) / sigma
        X_test_z = (X_test - mu) / sigma

        if classifier == 'logistic':
            from sklearn.linear_model import LogisticRegression
            if n_decode_classes == 2:
                clf = LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    max_iter=3000,
                    solver='lbfgs',
                    class_weight=class_weight,
                    random_state=random_state,
                )
            else:
                clf = LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    max_iter=3000,
                    multi_class='multinomial',
                    solver='lbfgs',
                    class_weight=class_weight,
                    random_state=random_state,
                )
            clf.fit(X_train_z, y_train)
            y_pred = clf.predict(X_test_z)
        elif classifier == 'svm_linear':
            from sklearn.svm import SVC
            clf = SVC(
                kernel='linear',
                C=svm_C,
                class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X_train_z, y_train)
            y_pred = clf.predict(X_test_z)
        elif classifier == 'svm_rbf':
            from sklearn.svm import SVC
            clf = SVC(
                kernel='rbf',
                C=svm_C,
                gamma=svm_gamma,
                class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X_train_z, y_train)
            y_pred = clf.predict(X_test_z)
        elif classifier == 'mlp':
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(
                hidden_layer_sizes=mlp_hidden_layer_sizes,
                alpha=mlp_alpha,
                max_iter=mlp_max_iter,
                random_state=random_state,
            )
            clf.fit(X_train_z, y_train)
            y_pred = clf.predict(X_test_z)
        elif classifier == 'nearest_centroid':
            centroids = np.stack(
                [X_train_z[y_train == lbl].mean(axis=0) for lbl in range(n_decode_classes)],
                axis=0
            )
            d2 = ((X_test_z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            y_pred = np.argmin(d2, axis=1)
        else:
            raise ValueError(
                "classifier must be one of: "
                "'logistic', 'nearest_centroid', 'svm_linear', 'svm_rbf', 'mlp'"
            )

        y_true_by_time.append(y_test)
        y_pred_by_time.append(y_pred)
        accuracy_by_time[t] = np.mean(y_pred == y_test)
        recalls = []
        for cls in range(n_decode_classes):
            cls_mask = (y_test == cls)
            if np.any(cls_mask):
                rec = np.mean(y_pred[cls_mask] == cls)
            else:
                rec = np.nan
            per_class_recall_by_time[t, cls] = rec
            recalls.append(rec)
        balanced_accuracy_by_time[t] = np.nanmean(recalls)
        for yi, pi in zip(y_test, y_pred):
            confusion_by_time[t, yi, pi] += 1

    return {
        'trial_types': trial_types,
        'class_names': class_names,
        'condition_to_class': {tt: class_names[lab] for tt, lab in zip(trial_types, cond_decode_labels)},
        'decode_mode': decode_mode,
        'usable_bins': usable_bins,
        'bin_edges': np.concatenate([[0], np.cumsum(lengths)]),
        'used_independent_test_split': X_test_full is not None,
        'accuracy_by_time': accuracy_by_time,
        'balanced_accuracy_by_time': balanced_accuracy_by_time,
        'per_class_recall_by_time': per_class_recall_by_time,
        'train_class_counts_by_time': train_class_counts_by_time,
        'chance_level': 1.0 / n_decode_classes,
        'y_true_by_time': np.array(y_true_by_time),
        'y_pred_by_time': np.array(y_pred_by_time),
        'confusion_by_time': confusion_by_time,
    }


def plot_decoding_model_comparison(
    results_by_model,
    mistake_class_name='mistake',
    figsize=(10, 3),
):
    """
    Compare time-resolved decoding metrics across multiple model result dicts.

    Parameters
    ----------
    results_by_model : dict[str, dict]
        Mapping from model label to output dict from
        ``decode_trial_conditions_over_time``.
    mistake_class_name : str, default='mistake'
        Class name used to select mistake recall from
        ``per_class_recall_by_time``.
    figsize : tuple, default=(10, 3)
        Figure size for the 2-panel comparison plot.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Left panel: balanced accuracy over time.
        Right panel: recall of ``mistake_class_name`` over time.
    """
    if not results_by_model:
        raise ValueError("results_by_model must be a non-empty dict")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    ax_bal, ax_mis = axes

    chance_values = []

    for model_name, res in results_by_model.items():
        if 'balanced_accuracy_by_time' not in res:
            raise KeyError(
                f"{model_name}: missing 'balanced_accuracy_by_time'. "
                "Use decode_trial_conditions_over_time outputs."
            )
        if 'per_class_recall_by_time' not in res or 'class_names' not in res:
            raise KeyError(
                f"{model_name}: missing 'per_class_recall_by_time' or 'class_names'."
            )

        bal = np.asarray(res['balanced_accuracy_by_time'])
        x = np.arange(bal.size)
        ax_bal.plot(x, bal, lw=2, label=model_name)

        class_names = list(res['class_names'])
        if mistake_class_name not in class_names:
            raise ValueError(
                f"{model_name}: class '{mistake_class_name}' not in class_names={class_names}"
            )
        mistake_idx = class_names.index(mistake_class_name)
        rec = np.asarray(res['per_class_recall_by_time'])[:, mistake_idx]
        ax_mis.plot(np.arange(rec.size), rec, lw=2, label=model_name)

        if 'chance_level' in res:
            chance_values.append(float(res['chance_level']))

    if chance_values:
        ax_bal.axhline(np.mean(chance_values), ls='--', c='k', alpha=0.5, label='chance')
    ax_mis.axhline(0.5, ls='--', c='k', alpha=0.5, label='0.5')

    ax_bal.set_title('Balanced Accuracy')
    ax_bal.set_xlabel('Time bin')
    ax_bal.set_ylabel('Score')
    ax_bal.set_ylim(0, 1)
    ax_bal.legend(frameon=False)

    ax_mis.set_title(f"{mistake_class_name} Recall")
    ax_mis.set_xlabel('Time bin')
    ax_mis.set_ylabel('Recall')
    ax_mis.set_ylim(0, 1)
    ax_mis.legend(frameon=False)

    fig.tight_layout()
    return fig, axes


def run_decoding_seed_stability(
    base_decode_kwargs,
    model_kwargs_by_name=None,
    n_seeds=10,
    seeds=None,
    mistake_class_name='mistake',
    consistency_reference_model='logistic',
    consistency_min_delta=0.0,
):
    """
    Run decoding repeatedly across random seeds and aggregate stability metrics.

    Parameters
    ----------
    base_decode_kwargs : dict
        Keyword args shared by all calls to ``decode_trial_conditions_over_time``
        (except ``random_state`` and model-specific kwargs).
    model_kwargs_by_name : dict[str, dict] | None, optional
        Mapping from model label to model-specific kwargs. If None, defaults to
        logistic/SVM/MLP comparison.
    n_seeds : int, default=10
        Number of seeds if ``seeds`` is None.
    seeds : sequence[int] | None, optional
        Exact seeds to evaluate. If None, uses ``range(n_seeds)``.
    mistake_class_name : str, default='mistake'
        Class name used to extract mistake recall from per-class recall outputs.
    consistency_reference_model : str, default='logistic'
        Model used as reference for consistency checks.
    consistency_min_delta : float, default=0.0
        Minimum per-seed improvement in mean mistake recall over reference model
        to count as a consistent improvement.

    Returns
    -------
    out : dict
        Contains seeds, per-seed model results, aggregate mean/std curves, and
        consistency summaries versus ``consistency_reference_model``.
    """
    if model_kwargs_by_name is None:
        model_kwargs_by_name = {
            'logistic': {'classifier': 'logistic'},
            'svm_linear': {'classifier': 'svm_linear'},
            'svm_rbf': {'classifier': 'svm_rbf'},
            'mlp': {'classifier': 'mlp'},
        }
    if seeds is None:
        seeds = list(range(int(n_seeds)))
    else:
        seeds = [int(s) for s in seeds]
    if len(seeds) == 0:
        raise ValueError("seeds must contain at least one value")

    per_model = {}
    for model_name, model_kwargs in model_kwargs_by_name.items():
        seed_results = []
        for seed in seeds:
            call_kwargs = dict(base_decode_kwargs)
            call_kwargs.update(model_kwargs)
            call_kwargs['random_state'] = seed
            res = decode_trial_conditions_over_time(**call_kwargs)
            seed_results.append(res)

        class_names = list(seed_results[0]['class_names'])
        if mistake_class_name not in class_names:
            raise ValueError(
                f"{model_name}: class '{mistake_class_name}' not in class_names={class_names}"
            )
        mistake_idx = class_names.index(mistake_class_name)

        bal_stack = np.stack(
            [np.asarray(r['balanced_accuracy_by_time']) for r in seed_results],
            axis=0
        )
        mistake_stack = np.stack(
            [np.asarray(r['per_class_recall_by_time'])[:, mistake_idx] for r in seed_results],
            axis=0
        )
        acc_stack = np.stack(
            [np.asarray(r['accuracy_by_time']) for r in seed_results],
            axis=0
        )

        per_model[model_name] = {
            'seed_results': seed_results,
            'balanced_accuracy': {
                'per_seed': bal_stack,
                'mean': np.nanmean(bal_stack, axis=0),
                'std': np.nanstd(bal_stack, axis=0),
                'mean_over_time_per_seed': np.nanmean(bal_stack, axis=1),
            },
            'mistake_recall': {
                'per_seed': mistake_stack,
                'mean': np.nanmean(mistake_stack, axis=0),
                'std': np.nanstd(mistake_stack, axis=0),
                'mean_over_time_per_seed': np.nanmean(mistake_stack, axis=1),
            },
            'accuracy': {
                'per_seed': acc_stack,
                'mean': np.nanmean(acc_stack, axis=0),
                'std': np.nanstd(acc_stack, axis=0),
                'mean_over_time_per_seed': np.nanmean(acc_stack, axis=1),
            },
            'class_names': class_names,
        }

    consistency = {}
    if consistency_reference_model in per_model:
        ref = per_model[consistency_reference_model]['mistake_recall']['mean_over_time_per_seed']
        for model_name, data in per_model.items():
            cur = data['mistake_recall']['mean_over_time_per_seed']
            delta = cur - ref
            consistency[model_name] = {
                'mean_delta_vs_reference': float(np.nanmean(delta)),
                'std_delta_vs_reference': float(np.nanstd(delta)),
                'fraction_seeds_improved': float(np.nanmean(delta > consistency_min_delta)),
                'reference_model': consistency_reference_model,
                'min_delta': float(consistency_min_delta),
            }

    return {
        'seeds': np.asarray(seeds),
        'models': per_model,
        'consistency': consistency,
        'mistake_class_name': mistake_class_name,
    }


def plot_decoding_seed_stability(
    stability_results,
    figsize=(11, 3.5),
    alpha_fill=0.2,
):
    """
    Plot mean +/- std curves from ``run_decoding_seed_stability`` output.

    Parameters
    ----------
    stability_results : dict
        Output from ``run_decoding_seed_stability``.
    figsize : tuple, default=(11, 3.5)
        Figure size.
    alpha_fill : float, default=0.2
        Transparency for std shading.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        Left panel: balanced accuracy mean +/- std.
        Right panel: mistake recall mean +/- std.
    """
    if 'models' not in stability_results:
        raise KeyError("stability_results must come from run_decoding_seed_stability")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False)
    ax_bal, ax_mis = axes

    for model_name, data in stability_results['models'].items():
        bal_mean = np.asarray(data['balanced_accuracy']['mean'])
        bal_std = np.asarray(data['balanced_accuracy']['std'])
        x_bal = np.arange(bal_mean.size)
        line_bal = ax_bal.plot(x_bal, bal_mean, lw=2, label=model_name)[0]
        c = line_bal.get_color()
        ax_bal.fill_between(
            x_bal,
            np.clip(bal_mean - bal_std, 0, 1),
            np.clip(bal_mean + bal_std, 0, 1),
            alpha=alpha_fill,
            color=c,
        )

        mis_mean = np.asarray(data['mistake_recall']['mean'])
        mis_std = np.asarray(data['mistake_recall']['std'])
        x_mis = np.arange(mis_mean.size)
        line_mis = ax_mis.plot(x_mis, mis_mean, lw=2, label=model_name)[0]
        c2 = line_mis.get_color()
        ax_mis.fill_between(
            x_mis,
            np.clip(mis_mean - mis_std, 0, 1),
            np.clip(mis_mean + mis_std, 0, 1),
            alpha=alpha_fill,
            color=c2,
        )

    ax_bal.set_title('Balanced Accuracy (mean +/- std)')
    ax_bal.set_xlabel('Time bin')
    ax_bal.set_ylabel('Score')
    ax_bal.set_ylim(0, 1)
    ax_bal.legend(frameon=False)

    mis_name = stability_results.get('mistake_class_name', 'mistake')
    ax_mis.set_title(f"{mis_name} Recall (mean +/- std)")
    ax_mis.set_xlabel('Time bin')
    ax_mis.set_ylabel('Recall')
    ax_mis.set_ylim(0, 1)
    ax_mis.legend(frameon=False)

    fig.tight_layout()
    return fig, axes
