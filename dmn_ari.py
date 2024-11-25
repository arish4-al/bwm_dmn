from one.api import ONE
from brainbox.singlecell import bin_spikes2D
from brainwidemap import (bwm_query, load_good_units, 
                          load_trials_and_mask, bwm_units)
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import iblatlas
from iblatlas.plots import plot_swanson_vector 
from brainbox.io.one import SessionLoader

import sys
sys.path.append('Dropbox/scripts/IBL/')

from scipy import signal
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from sklearn.cluster import DBSCAN, OPTICS, Birch, MiniBatchKMeans
from numpy.linalg import norm
from scipy.stats import gaussian_kde, f_oneway, pearsonr, spearmanr, kruskal
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, cdist
from sklearn.preprocessing import StandardScaler
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sknetwork.clustering import Louvain, Leiden, KCenters
from sknetwork.visualization import visualize_graph
from IPython.display import SVG
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy import sparse
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
import gc
from pathlib import Path
import random
from copy import deepcopy
import time, sys, math, string, os
from scipy.stats import spearmanr, zscore, linregress
import umap
from itertools import combinations, chain
from datetime import datetime
import scipy.ndimage as ndi
#import hdbscan

from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap   
from matplotlib.gridspec import GridSpec   
import mpldatacursor
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import cm
from venny4py.venny4py import *
import networkx as nx
from termcolor import colored

import warnings
warnings.filterwarnings("ignore")
#mpl.use('QtAgg')

# for vari plot
#_, b, lab_cols = labs()
plt.ion() 
 
np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update(plt.rcParamsDefault)
plt.ion()

f_size = 15  # font size

# canonical colors for left and right trial types
blue_left = [0.13850039, 0.41331206, 0.74052025]
red_right = [0.66080672, 0.21526712, 0.23069468]

T_BIN = 0.0125  # bin size [sec] for neural binning
sts = 0.002  # stride size in [sec] for overlapping bins
ntravis = 30  # #trajectories for vis, first 2 real, rest pseudo

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', silent=True)

#base_url='https://openalyx.internationalbrainlab.org',
#          password='international', silent=True 
                   
br = BrainRegions()
#units_df = bwm_units(one)  # canonical set of cells


# save results here
pth_dmn = Path(one.cache_dir, 'dmn', 'res')
pth_dmn.mkdir(parents=True, exist_ok=True)

sigl = 0.05  # significance level (for stacking, plotting, fdr)


# order sensitive: must be tts__ = concat_PETHs(pid, get_tts=True).keys()
tts__ = ['inter_trial', 'blockL', 'blockR', 'block50', 'quiescence', 'stimLbLcL', 'stimLbRcL', 'stimLbRcR', 'stimLbLcR', 'stimRbLcL', 'stimRbRcL', 'stimRbRcR', 'stimRbLcR', 'motor_init', 'sLbLchoiceL', 'sLbRchoiceL', 'sLbRchoiceR', 'sLbLchoiceR', 'sRbLchoiceL', 'sRbRchoiceL', 'sRbRchoiceR', 'sRbLchoiceR', 'choiceL', 'choiceR',  'fback1', 'fback0']
#'fback0sRbL', 'fback0sLbR',


PETH_types_dict = {
    'concat': [item for item in tts__],
    'resting': ['inter_trial'],
    'quiescence': ['quiescence'],
    'pre-stim-prior': ['blockL', 'blockR'],
    'block50': ['block50'],
    'stim_all': ['stimLbRcL','stimRbLcR','stimLbLcL', 'stimRbRcR'],
    'stim_surp_incon': ['stimLbRcL','stimRbLcR'],
    'stim_surp_con': ['stimLbLcL', 'stimRbRcR'],
    #'resp_surp': ['fback0sRbL', 'fback0sLbR'],
    'motor_init': ['motor_init'],
    'fback1': ['fback1'],
    'fback0': ['fback0']}  


cortical_regions = {
    "Prefrontal": [
        "FRP", "ACAd", "ACAv", "PL", "ILA",
        "ORBl", "ORBm", "ORBvl"
    ],
    "Lateral": [
        "AId", "AIv", "AIp", "GU", "VISc",
        "TEa", "PERI", "ECT"
    ],
    "Somatomotor": [
        "SSs", "SSp-bfd", "SSp-tr", "SSp-ll",
        "SSp-ul", "SSp-un", "SSp-n", "SSp-m",
        "MOp", "MOs"
    ],
    "Visual": [
        "VISal", "VISl", "VISp", "VISpl",
        "VISli", "VISpor", "VISrl"
    ],
    "Medial": [
        "VISa", "VISam", "VISpm",
        "RSPagl", "RSPd", "RSPv"
    ],
    "Auditory": [
        "AUDd", "AUDp", "AUDpo", "AUDv"
    ]
}

cortical_colors = {"Prefrontal": 'r', 
                   "Lateral": 'yellow',
                   "Somatomotor": 'orange',
                   "Visual": 'g',
                   "Medial": 'blue',
                   "Auditory": 'purple'
                  }


dmn_regs = ['ACAd', 'ACAv', 'PL', 'ILA', 'ORBl', 'ORBm', 
            'ORBvl', 'VISa', 'VISam', 'RSPagl','RSPd', 
            'RSPv', 'SSp-tr', 'SSp-ll', 'MOs']


def put_panel_label(ax, k):
    ax.annotate(string.ascii_lowercase[k], (-0.05, 1.15),
                xycoords='axes fraction',
                fontsize=f_size * 1.5, va='top',
                ha='right', weight='bold')


def grad(c, nobs, fr=1):
    '''
    color gradient for plotting trajectories
    c: color map type
    nobs: number of observations
    '''

    cmap = mpl.cm.get_cmap(c)

    return [cmap(fr * (nobs - p) / nobs) for p in range(nobs)]


def get_name(brainregion):
    '''
    get verbose name for brain region acronym
    '''
    regid = br.id[np.argwhere(br.acronym == brainregion)][0, 0]
    return br.name[np.argwhere(br.id == regid)[0, 0]]


def get_allen_info(rerun=False):
    '''
    Function to load Allen atlas info, like region colors
    '''
    
    pth_dmna = Path(one.cache_dir, 'dmn', 'alleninfo.npy')
    
    if (not pth_dmna.is_file() or rerun):
        p = (Path(ibllib.__file__).parent /
             'atlas/allen_structure_tree.csv')

        dfa = pd.read_csv(p)

        # replace yellow by brown #767a3a    
        cosmos = []
        cht = []
        
        for i in range(len(dfa)):
            try:
                ind = dfa.iloc[i]['structure_id_path'].split('/')[4]
                cr = br.id2acronym(ind, mapping='Cosmos')[0]
                cosmos.append(cr)
                if cr == 'CB':
                    cht.append('767A3A')
                else:
                    cht.append(dfa.iloc[i]['color_hex_triplet'])    
                        
            except:
                cosmos.append('void')
                cht.append('FFFFFF')
                

        dfa['Cosmos'] = cosmos
        dfa['color_hex_triplet2'] = cht
        
        # get colors per acronym and transfomr into RGB
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].fillna('FFFFFF')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].replace('19399', '19399a')
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'].replace(
                                                         '0', 'FFFFFF')
        dfa['color_hex_triplet2'] = '#' + dfa['color_hex_triplet2'].astype(str)
        dfa['color_hex_triplet2'] = dfa['color_hex_triplet2'
                                       ].apply(lambda x:
                                               mpl.colors.to_rgba(x))

        palette = dict(zip(dfa.acronym, dfa.color_hex_triplet2))

        #add layer colors
        bc = ['b', 'g', 'r', 'c', 'm', 'y', 'brown', 'pink']
        for i in range(7):
            palette[str(i)] = bc[i]
        
        palette['thal'] = 'k'    
        r = {}
        r['dfa'] = dfa
        r['palette'] = palette    
        np.save(pth_dmna, r, allow_pickle=True)   

    r = np.load(pth_dmna, allow_pickle=True).flat[0]
    return r['dfa'], r['palette']  



def cosine_sim(v0, v1):
    # cosine similarity 
    return np.inner(v0,v1)/ (norm(v0) * norm(v1))



def get_reg_dist(rerun=False, algo='umap_z', 
                  mapping='Beryl', vers='concat'):

    pth_ = Path(one.cache_dir, 'dmn', 
                f'{algo}_{mapping}_{vers}_smooth.npy')
    if (not pth_.is_file() or rerun):
        res, regs = smooth_dist(algo=algo, mapping=mapping, vers=vers)    
        d = {'res': res, 'regs' : regs}
        np.save(pth_, d, allow_pickle=True)
    else:
        d = np.load(pth_, allow_pickle=True).flat[0]        
        
    return d     



def clustering_on_peth_data(r, algo='concat_z', k=2, clustering='kmeans', min_s=10, eps=0.5, random_state=0):

    res = r[algo]
    
    if clustering=='hierarchy': # Order the matrix using hierarchical clustering            
        linkage_matrix = hierarchy.linkage(res)
        #ordered_indices = hierarchy.leaves_list(linkage_matrix)            
        clusters = hierarchy.fcluster(linkage_matrix, k, criterion='maxclust')

    elif clustering == 'spectralco': # Order the matrix using spectral co-clustering
        clustering_result = SpectralCoclustering(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.row_labels_

    elif clustering == 'spectralbi': #spectral bi-clustering
        clustering_result = SpectralBiclustering(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.row_labels_

    elif clustering == 'dbscan':
        clustering_result = DBSCAN(eps=eps, min_samples=min_s, metric='cosine').fit(res)
        clusters = clustering_result.labels_
        
    elif clustering == 'birch': #birch clustering
        clustering_result = Birch(n_clusters=k).fit(res)
        clusters = clustering_result.labels_
        
    elif clustering == 'mbkmeans': 
        clustering_result = MiniBatchKMeans(n_clusters=k, batch_size=20, random_state=random_state).fit(res)
        clusters = clustering_result.labels_

    elif clustering == 'kmeans': 
        clustering_result = KMeans(n_clusters=k, random_state=random_state).fit(res)
        clusters = clustering_result.labels_
        r['centers'] = clustering_result.cluster_centers_
        #print(clustering_result.inertia_)

    else:
        print('what clustering method?')
        return
    
    return r, clusters


def regional_group(mapping, algo, vers='concat', norm_=False, min_s=10, eps=0.5,
                   run_umap=False, n_neighbors=10, d=0.2, ncomp=2,
                   nclus = 7, random_seed = 0):

    '''
    find group labels for all cells
    mapping: [Allen, Beryl, Cosmos, layers, clusters, clusters_xyz] or some clustering algorithm name
    algo: concat_z(original high-dim data) or umap_z(2d dim-reduced data)
    '''

    r = np.load(Path(pth_dmn, f'{vers}_norm{norm_}.npy'),
                 allow_pickle=True).flat[0]
                 
                              
    if run_umap==True:
        r['umap_z'] = umap.UMAP(random_state=random_seed, n_components=ncomp, min_dist=d, 
                             n_neighbors=n_neighbors).fit_transform(r['concat_z'])

    if algo=='pca_z':
        r['pca_z'] = PCA(n_components=ncomp).fit_transform(r['concat_z'])
                   

    # add point names to dict
    r['nums'] = range(len(r[algo][:,0]))

    if mapping in ['kmeans', 'mbkmeans', 'dbscan', 'hierarchy', 'birch', 'spectralbi', 'spectralco']:
        # use the corresponding clustering method on full data or dim-reduced 2d data
        r, clusters = clustering_on_peth_data(r, algo=algo, k=nclus, 
                                              clustering=mapping, 
                                              min_s=min_s, eps=eps, 
                                              random_state=random_seed) 
               
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters
        regs = np.unique(clusters)
            
        color_map = dict(zip(list(acs), list(cols)))
        r['els'] = [Line2D([0], [0], color=color_map[reg], 
                    lw=4, label=f'{reg + 1}')
                    for reg in regs]
        
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}
              

    elif mapping == 'hdbscan':
        mcs = 10
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)    
        clusterer.fit(r[algo])
        labels = clusterer.labels_
        unique_labels = np.unique(labels)
        mapping = {old_label: new_label 
                      for new_label, old_label in 
                      enumerate(unique_labels)}
        clusters = np.array([mapping[label] for label in labels])

        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/len(unique_labels))
        acs = clusters
        # get average point and color per region
        av = {clus: [np.mean(r[algo][clusters == clus], axis=0), 
                    cols] 
              for clus in range(1,len(unique_labels)+1)} 
        


    elif mapping == 'layers':       
    
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping='Allen'))
        
        regs0 = Counter(acs)
                                     
        # get regs with number at and of acronym
        regs = [reg for reg in regs0 
                if reg[-1].isdigit()]
        
        for reg in regs:        
            acs[acs == reg] = reg[-1]       
        
        # extra class of thalamic (and hypothalamic) regions 
        names = dict(zip(regs0,[get_name(reg) for reg in regs0]))
        thal = {x:names[x] for x in names if 'thala' in names[x]}
                                          
        for reg in thal: 
            acs[acs == reg] = 'thal'       
        
        mask = np.array([(x.isdigit() or x == 'thal') for x in acs])
        acs[~mask] = '0'
        
        remove_0 = True
        
        if remove_0:
            # also remove layer 6, as there are only 20 neurons 
            zeros = np.arange(len(acs))[
                        np.bitwise_or(acs == '0', acs == '6')]
            for key in r:
                if len(r[key]) == len(acs):
                    r[key] = np.delete(r[key], zeros, axis=0)
                       
            acs = np.delete(acs, zeros)        
        
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        regs = Counter(acs)      
        r['els'] = [Line2D([0], [0], color=pa[reg], 
               lw=4, label=f'{reg} {regs[reg]}')
               for reg in regs]
               
        # get average points and color per region
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
               

    elif mapping == 'clusters_xyz':
   
        # use clusters from hierarchical clustering to color
        nclus = 1000
        clusters = fcluster(r['linked_xyz'], t=nclus, 
                            criterion='maxclust')
        cmap = mpl.cm.get_cmap('Spectral')
        cols = cmap(clusters/nclus)
        acs = clusters   
        # get average points per region
        av = {reg: [np.mean(r[algo][clusters == clus], axis=0), 
                    cmap(clus/nclus)] 
              for clus in range(1,nclus+1)}      

    else:
        acs = np.array(br.id2acronym(r['ids'], 
                                     mapping=mapping))
                                     
#        # remove void and root
#        zeros = np.arange(len(acs))[np.bitwise_or(acs == 'root',
#                                                  acs == 'void')]
#        for key in r:
#            if len(r[key]) == len(acs):
#                r[key] = np.delete(r[key], zeros, axis=0)
#                   
#        acs = np.delete(acs, zeros)          
        
                                                              
        _,pa = get_allen_info()
        cols = [pa[reg] for reg in acs]
        
        # get average points and color per region
        regs = Counter(acs)  
        av = {reg: [np.mean(r[algo][acs == reg], axis=0), pa[reg]] 
              for reg in regs}
              

    if 'end' in r['len']:
        del r['len']['end']
              
    r['acs'] = acs
    r['cols'] = cols
    r['av'] = av

    return r



def smooth_dist(algo='umap_z', mapping='Beryl', show_imgs=False,
                norm_=True, dendro=True, nmin=30, vers='concat'):

    '''
    smooth 2d pointclouds, show per class
    norm_: normalize smoothed image by max brightness
    '''

    r = regional_group(mapping, algo, vers=vers)
    feat = 'concat_z' if algo[-1] == 'z' else 'concat'
    fontsize = 12
    
    # Define grid size and density kernel size
    x_min = np.floor(np.min(r[algo][:,0]))
    x_max = np.ceil(np.max(r[algo][:,0]))
    y_min = np.floor(np.min(r[algo][:,1]))
    y_max = np.ceil(np.max(r[algo][:,1]))
    
    imgs = {}
    xys = {}
    
    regs00 = Counter(r['acs'])
    regcol = {reg: np.array(r['cols'])[r['acs'] == reg][0] 
              for reg in regs00}    

    if mapping == 'Beryl':
        # oder regions 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regsord = dict(zip(br.id2acronym(np.load(p), 
                           mapping='Beryl'),
                           br.id2acronym(np.load(p), 
                           mapping='Cosmos')))
        regs = []
        
        for reg in regsord:
            if ((reg in regs00) and (regs00[reg] > nmin)):
                regs.append(reg)
    
    else:
        regs = [reg for reg in regs00 if 
                regs00[reg] > nmin]

    for reg in regs:
    
        # scale values to lie within unit interval
        x = (r[algo][np.array(r['acs'])==reg,0] - x_min)/ (x_max - x_min)    
        y = (r[algo][np.array(r['acs'])==reg,1] - y_min)/ (y_max - y_min)

        data = np.array([x,y]).T         
        inds = (data * 255).astype('uint')  # convert to indices

        img = np.zeros((256,256))  # blank image
        for i in np.arange(data.shape[0]):  # draw pixels
            img[inds[i,0], inds[i,1]] += 1
        
        imsm = ndi.gaussian_filter(img.T, (10,10))
        imgs[reg] = imsm/np.max(imsm) if norm_ else imsm
        xys[reg] = [x,y]
  

    if show_imgs:

        # tweak for other mapping than "layers"
        fig, axs = plt.subplots(nrows=3, ncols=len(regs),
                                figsize=(18.6, 5.8))        
        axs = axs.flatten()    
        #[ax.set_axis_off() for ax in axs]

        vmin = np.min([np.min(imgs[reg].flatten()) for reg in imgs])
        vmax = np.max([np.max(imgs[reg].flatten()) for reg in imgs])
        
        k = 0 

        # row of images showing point clouds     
        for reg in imgs:
            axs[k].scatter(xys[reg][0], xys[reg][1], color=regcol[reg], s=0.1)
            axs[k].set_title(f'{reg}, ({regs00[reg]})')
            #axs[k].set_axis_off()
            axs[k].set_aspect('equal')
            axs[k].spines['right'].set_visible(False)
            axs[k].spines['top'].set_visible(False)
            axs[k].set_xlabel('umap dim 1')
            axs[k].set_ylabel('umap dim 2')             
            k+=1
            
        # row of panels showing smoothed point clouds
        for reg in imgs:
            axs[k].imshow(imgs[reg], origin='lower', vmin=vmin, vmax=vmax,
                          interpolation=None)
            axs[k].set_title(f'{reg}, ({regs00[reg]})')
            axs[k].set_axis_off()
            k+=1                            
            
        # row of images showing mean feature vector
        for reg in imgs:
            pts = np.arange(len(r['acs']))[r['acs'] == reg]
            
            xss = T_BIN * np.arange(len(np.mean(r[feat][pts],axis=0)))
            yss = np.mean(r[feat][pts],axis=0)
            yss_err = np.std(r[feat][pts],axis=0)/np.sqrt(len(pts))
                         
            axs[k].fill_between(xss, yss - yss_err, yss + yss_err, 
                                alpha=0.2, color = regcol[reg])    
                
            maxys = [yss + yss_err]  
              
            #region mean
            axs[k].plot(xss,yss, color='k', linewidth=2)    

            axs[k].set_title(reg)
            axs[k].set_xlabel('time [sec]')
            axs[k].set_ylabel(feat)
            axs[k].set_axis_off()      
        
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axs[k].axvline(T_BIN * xv, linestyle='--',
                            color='grey', linewidth=0.1)
                            
                axs[k].text(T_BIN * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=5, color='k')
            
                h += r['len'][i]
            
            k+=1
            
        
        fig.suptitle(f'algo: {algo}, mapping: {mapping}, norm:{norm_}')
        fig.tight_layout()    

    # show cosine similarity of density vectors
    

    
    res = np.zeros((len(regs),len(regs)))
    i = 0
    for reg_i in imgs:
        j = 0
        for reg_j in imgs:
            v0 = imgs[reg_i].flatten()
            v1 = imgs[reg_j].flatten()
            
            res[i,j] = cosine_sim(v0, v1)
            j+=1
        i+=1            

    if dendro:
        fig0, axs = plt.subplots(ncols=2, figsize=(10,8), 
            gridspec_kw={'width_ratios': [1, 11]})
        res = np.round(res, decimals=8)
        
        cres = squareform(1 - res)
        linkage_matrix = hierarchy.linkage(cres)
        

        # Order the matrix using the hierarchical clustering
        ordered_indices = hierarchy.leaves_list(linkage_matrix)
        res = res[:, ordered_indices][ordered_indices, :]
        
        row_dendrogram = hierarchy.dendrogram(linkage_matrix,labels =regs,
                     orientation="left", color_threshold=np.inf, ax=axs[0])
        regs = np.array(regs)[ordered_indices]
        
        [t.set_color(i) for (i,t) in    
            zip([regcol[reg] for reg in regs],
                 axs[0].yaxis.get_ticklabels())]
                                     
                     
        ax0 = axs[1]
        
        axs[0].axis('off')
#        axs[0].tick_params(axis='both', labelsize=fontsize)
#        axs[0].spines['top'].set_visible(False)
#        axs[0].spines['bottom'].set_visible(False)    
#        axs[0].spines['right'].set_visible(False)
#        axs[0].spines['left'].set_visible(False)
#        axs[0].set_xticks([])
        
        
    else:
        fig0, ax0 = plt.subplots(figsize=(4,4))
    
                   
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(regs)), regs,
                   rotation=90, fontsize=fontsize)
    ax0.set_yticks(np.arange(len(regs)), regs, fontsize=fontsize)               
                   
    [t.set_color(i) for (i,t) in
        zip([regcol[reg] for reg in regs],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([regcol[reg] for reg in regs],
        ax0.yaxis.get_ticklabels())]
    
    #ax0.set_title(f'cosine similarity of smooth images, norm:{norm_}')
    #ax0.set_ylabel(mapping)
    cb = plt.colorbar(ims,fraction=0.046, pad=0.04)
    cb.set_label('regional similarity')
    fig0.tight_layout()
    #fig0.suptitle(f'{algo}, {mapping}')
    
    return res, regs



def compare_distance_metrics_scatter(vers, nclus=7, nd=2, rerun=False):
    # load data
    wass = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), allow_pickle=True).flat[0]
    wass_d = wass['res']
    wass_regs = wass['regs']
    umap = get_reg_dist(algo='umap_z', vers=vers, rerun=rerun)
    umap_d = umap['res']
    umap_regs = umap['regs']

    # reorder umap similarity matrix w/ wasserstein matrix entries' ordering
    ordered_umap_indices = [list(umap_regs).index(reg) for reg in wass_regs]
    umap_regs = np.array(umap_regs)[ordered_umap_indices]
    umap_d = umap_d[:, ordered_umap_indices][ordered_umap_indices, :]

    umap_d_flat = umap_d.flatten()
    wass_d_flat = wass_d.flatten()

    corp,pp = pearsonr(umap_d_flat, wass_d_flat)
    cors,ps = spearmanr(umap_d_flat, wass_d_flat)

    print(corp, pp, cors, ps)

    plt.scatter(umap_d_flat, wass_d_flat, s=2)
    plt.xlabel('Umap Cosine Similarity')
    plt.ylabel('Wasserstein Distance')
    plt.title(f'{vers}\n pearsonr: {corp}, spearmanr: {cors}')
    plt.savefig(Path(pth_dmn.parent, 'figs', f'compare_metrics_{vers}.pdf'))
    plt.show()


def clustering_on_connectivity_matrix(res, regs, k=2, metric='umap_z', clustering='hierarchy', 
                                      random_state=0, resl=1.01):
    if clustering=='hierarchy': # Order the matrix using hierarchical clustering
        
        if metric in ['umap_z', 'pca_z']: # convert similarity scores to distances
            res0 = np.copy(res)
            res = np.amax(res) - res
            
        # get condensed distance matrix (upper-triangular part of distance matrix) as input
        cres = squareform(res)
        linkage_matrix = hierarchy.linkage(cres)
        ordered_indices = hierarchy.leaves_list(linkage_matrix)            
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        
        if metric in ['umap_z', 'pca_z']: # convert distances back to similarity scores
            res = res0
        elif metric == 'wass': # convert wasserstein distance to similarity for plotting
            res = (np.amax(res) - res)/np.amax(res)
            
        res = res[:, ordered_indices][ordered_indices, :]
        cluster_info = hierarchy.fcluster(linkage_matrix, 10, criterion='maxclust')

    elif clustering == 'spectralco': # Order the matrix using spectral co-clustering
        clustering_result = SpectralCoclustering(n_clusters=k, random_state=random_state).fit(res)
        ordered_indices_r = np.argsort(clustering_result.row_labels_)
        ordered_indices_c = np.argsort(clustering_result.column_labels_)
        regs_r = np.array(regs)[ordered_indices_r]
        regs_c = np.array(regs)[ordered_indices_c]
        res = res[ordered_indices_r]
        res = res[:, ordered_indices_c]
        cluster_info = clustering_result.row_labels_

    elif clustering == 'spectralbi': #spectral bi-clustering
        clustering_result = SpectralBiclustering(n_clusters=k, random_state=random_state).fit(res)
        ordered_indices_r = np.argsort(clustering_result.row_labels_)
        ordered_indices_c = np.argsort(clustering_result.column_labels_)
        regs_r = np.array(regs)[ordered_indices_r]
        regs_c = np.array(regs)[ordered_indices_c]
        res = res[ordered_indices_r]
        res = res[:, ordered_indices_c]
        cluster_info = clustering_result.row_labels_
        
    elif clustering == 'birch': #birch clustering
        clustering_result = Birch(n_clusters=k).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :] 
        
    elif clustering == 'mbkmeans': 
        clustering_result = MiniBatchKMeans(n_clusters=k, batch_size=20, random_state=random_state).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'kmeans': 
        clustering_result = KMeans(n_clusters=k, random_state=random_state).fit(res)
        cluster_info = clustering_result.labels_
        ordered_indices = np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'louvain':
        adjacency = sparse.csr_matrix(res)
        louvain = Louvain(random_state=random_state, resolution=resl)
        cluster_info = louvain.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'leiden':
        adjacency = sparse.csr_matrix(res)
        leiden = Leiden(random_state=random_state, resolution=resl)
        cluster_info = leiden.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering == 'kcenters':
        adjacency = sparse.csr_matrix(res)
        kcenters = KCenters(n_clusters=k)
        cluster_info = kcenters.fit_predict(adjacency)
        ordered_indices=np.argsort(cluster_info)
        regs_r = np.array(regs)[ordered_indices]
        regs_c = regs_r
        res = res[:, ordered_indices][ordered_indices, :]

    else:
        print('what clustering method?')
        return

    return res, regs_r, regs_c, cluster_info



def get_reproducibility_score(vers, clustering, k=None, resl=None):
    d = get_reg_dist(algo='umap_z', vers=vers)
    res = d['res']
    regs = d['regs']

    ARI, AMI = [], []
    for i in range(20):
        _, _, _, cluster_info0 = clustering_on_connectivity_matrix(
            res, regs, k=k, clustering=clustering, random_state=i, resl=resl)
        _, _, _, cluster_info1 = clustering_on_connectivity_matrix(
            res, regs, k=k, clustering=clustering, random_state=457-i, resl=resl)
    
        ARI.append(adjusted_rand_score(cluster_info0, cluster_info1))
        AMI.append(adjusted_mutual_info_score(cluster_info0, cluster_info1))
    print('mean ARI', np.mean(ARI), 'mean AMI', np.mean(AMI))


def get_quiescence_resting_diff(metric='umap_z', diff='shifted'):
    dr = get_reg_dist(algo=metric, vers='resting')
    dq = get_reg_dist(algo=metric, vers='quiescence')
    d = {}

    # order regions by canonical list 
    p = (Path(iblatlas.__file__).parent / 'beryl.npy')
    regs_can = br.id2acronym(np.load(p), mapping='Beryl')
    regs = [reg for reg in regs_can if reg in dr['regs']]
    ordered_indices_r = [list(dr['regs']).index(reg) for reg in regs]
    ordered_indices_q = [list(dq['regs']).index(reg) for reg in regs]
    res_r = dr['res'][:, ordered_indices_r][ordered_indices_r, :]
    res_q = dq['res'][:, ordered_indices_q][ordered_indices_q, :]

    if diff == 'shifted':
        d['res'] = res_q-res_r - np.min(res_q-res_r)
    else:
        d['res'] = abs(res_q-res_r)
    d['regs'] = regs

    return d



'''
plotting
'''


def plot_dim_reduction(algo_data='concat_z', algo='umap_z', mapping='Beryl', norm_=False,
                       run_umap=False, ncomp=2, n_neighbors=15, d=0.1,
                       min_s=10, eps=0.5, 
                       means=False, exa=False, shuf=False,
                       exa_squ=False, vers='concat', ax=None, ds=0.5,
                       axx=None, exa_clus=False, leg=False, restr=None,
                       nclus = 7, random_seed=0):
                       
    '''
    2 dims being pca on concat PETH; 
    colored by region
    algo_data in ['umap_z', 'concat_z']: chooses in which space to do clustering (method specified by mapping argument)
    algo in ['umap','tSNE','PCA','ICA']: in what space to plot data
    means: plot average dots per region
    exa: plot some example feature vectors
    exa_squ: highlight example squares in embedding space,
             make and extra plot for each with mean feature vector 
             and those of cells in square in color of mapping
    space: 'concat'  # can also be tSNE, PCA, umap, for distance space
    ds: marker size in main scatter
    restr: list of Beryl regions to restrict plot to
    '''
    
    feat = 'concat_z'
    
    r = regional_group(mapping, algo_data, vers=vers, norm_=norm_, min_s=min_s,
                       eps=eps, run_umap=run_umap, n_neighbors=n_neighbors, d=d,
                       ncomp=ncomp, nclus=nclus, random_seed=random_seed)
    alone = False
    if not ax:
        alone = True
        if ncomp==3:
            fig = plt.figure(dpi=200)
            ax = fig.add_subplot(projection='3d')
        else:
            fig, ax = plt.subplots(label=f'{vers}_{mapping}', dpi=200)
        #ax.set_title(vers)
    
    if shuf:
        shuffle(r['cols'])
    
    if restr:
        # restrict to certain Beryl regions
        #r2 = regional_group('Beryl', algo, vers=vers)
        ff = np.bitwise_or.reduce([r['acs'] == reg for reg in restr]) 
    
    
        im = ax.scatter(r[algo][:,0][ff], r[algo][:,1][ff], 
                        marker='o', c=r['cols'][ff], s=ds, rasterized=True)
                        
    else: 
        if ncomp==3:
            im = ax.scatter(r[algo][:,0], r[algo][:,1], r[algo][:,2],
                            marker='o', c=r['cols'], s=ds, rasterized=True)
        else:
            im = ax.scatter(r[algo][:,0], r[algo][:,1], 
                            marker='o', c=r['cols'], s=ds, rasterized=True)                            
                        
    
    if means:
        # show means
        emb1 = [r['av'][reg][0][0] for reg in r['av']]
        emb2 = [r['av'][reg][0][1] for reg in r['av']]
        cs = [r['av'][reg][1] for reg in r['av']]
        ax.scatter(emb1, emb2, marker='o', facecolors='none', 
                   edgecolors=cs, s=600, linewidths=4, rasterized=True)
    
#    ax.set_xlabel(f'{algo} dim1')
#    ax.set_ylabel(f'{algo} dim2')
    zs = True if algo == 'umap_z' else False
    if alone:
        ax.set_title(f'norm: {norm_}, z-score: {zs}')
    ax.axis('off')
    ss = 'shuf' if shuf else ''
       
    
    if mapping in ['layers', 'kmeans']:
        if leg:
            ax.legend(handles=r['els'], ncols=1,
                      frameon=False).set_draggable(True)

    elif 'clusters' in mapping:
        nclus = len(Counter(r['acs']))
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.01])
        norm = mpl.colors.Normalize(vmin=0, 
                                    vmax=nclus)
        cmap = mpl.cm.get_cmap('Spectral')                            
        fig.colorbar(mpl.cm.ScalarMappable(
                                norm=norm, 
                                cmap=cmap), 
                                cax=cax, orientation='horizontal')

    if alone:
        fig.tight_layout()
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
        f'{algo}_{vers}_{mapping}_{nclus}_{algo_data}_{ncomp}d_{d}_{n_neighbors}.pdf'), dpi=200, bbox_inches='tight')


    if exa:
        # plot a cells' feature vector
        # in extra panel when hovering over point
        fig_extra, ax_extra = plt.subplots()
        
        line, = ax_extra.plot(r[feat][0], 
                              label='Extra Line Plot')

        # Define a function to update the extra line plot 
        # based on the selected point
        
        def update_line(event):
            if event.mouseevent.inaxes == ax:
                x_clicked = event.mouseevent.xdata
                y_clicked = event.mouseevent.ydata
                
                selected_point = None
                for key, value in zip(r['nums'], r[algo]):
                    if (abs(value[0] - x_clicked) < 0.01 and 
                       abs(value[1] - y_clicked) < 0.01):
                        selected_point = key
                        break
                
                if selected_point:

                    line.set_data(T_BIN *np.arange(len(r[feat][key])),
                                  r[feat][key])
                    ax_extra.relim()
                    ax_extra.set_ylabel(feat)
                    ax_extra.set_xlabel('time [sec]')
                    ax_extra.autoscale_view()              
                    ax_extra.set_title(
                        f'Line Plot for x,y ='
                        f' {np.round(x_clicked,2), np.round(y_clicked,2)}')
                    fig_extra.canvas.draw()   
    
        # Connect the pick event to the scatter plot
        fig.canvas.mpl_connect('pick_event', update_line)
        im.set_picker(5)  # Set the picker radius for hover detection

    if exa_clus:
        # show for each cluter the mean PETH
        if axx is None:
            fg, axx = plt.subplots(nrows=len(np.unique(r['acs'])),
                                   sharex=True, sharey=False,
                                   figsize=(6,6))
                
        maxys = [np.max(np.mean(r[feat][
                 np.where(r['acs'] == clu)], axis=0)) 
                 for clu in np.unique(r['acs'])]
        
        kk = 0             
        for clu in np.unique(r['acs']):
                    
            #cluster mean
            xx = np.arange(len(r[feat][0])) /480
            yy = np.mean(r[feat][np.where(r['acs'] == clu)], axis=0)

            axx[kk].plot(xx, yy,
                     color=r['cols'][np.where(r['acs'] == clu)][0],
                     linewidth=2)
                     

            
            if kk != (len(np.unique(r['acs'])) - 1):
                axx[kk].axis('off')
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['left'].set_visible(False)      
                axx[kk].tick_params(left=False, labelleft=False)
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                axx[kk].axvline(xv/480, linestyle='--', linewidth=1,
                            color='grey')
                
                if  kk == 0:            
                    axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                             '   '+i, rotation=90, color='k', 
                             fontsize=10, ha='center')
            
                h += d2[i] 
            kk += 1                

#        #axx.set_title(f'{s} \n {len(pts)} points in square')
        axx[kk - 1].set_xlabel('time [sec]')
#        axx.set_ylabel(feat)
        if alone:
            fg.tight_layout()
        fg.savefig(Path(one.cache_dir,'dmn', 'figs',
            f'{vers}_{mapping}_clusters_{nclus}_{algo_data}.pdf'), dpi=150, bbox_inches='tight')


    if exa_squ:
    
        # get squares
        ns = 10  # number of random square regions of interest
        ss = 0.01  # square side length as a fraction of total area
        x_min = np.floor(np.min(r[algo][:,0]))
        x_max = np.ceil(np.max(r[algo][:,0]))
        y_min = np.floor(np.min(r[algo][:,1]))
        y_max = np.ceil(np.max(r[algo][:,1]))
        
        
        side_length = ss * (x_max - x_min)
        
        sqs = []
        for _ in range(ns):
            # Generate random x and y coordinates within the data range
            x = random.uniform(x_min, x_max - side_length)
            y = random.uniform(y_min, y_max - side_length)
            
            # Create a square represented as (x, y, side_length)
            square = (x, y, side_length)
            
            # Add the square to the list of selected squares
            sqs.append(square)
            

        
        r['nums'] = range(len(r[algo][:,0]))
        
        
        k = 0
        for s in sqs:
    
            
            # get points within square
            
            pts = []
            sq_x, sq_y, side_length = s
            
            for ke, value in zip(r['nums'], r[algo]):
                if ((sq_x <= value[0] <= sq_x + side_length) 
                    and (sq_y <= value[1] <= sq_y + side_length)):
                    pts.append(ke)            
          
            if len(pts) == 0:
                continue
          
            # plot squares in main figure
            rect = plt.Rectangle((s[0], s[1]), s[2], s[2], 
                    fill=False, color='r', linewidth=2)
            ax.add_patch(rect)
          
          
            # plot mean and individual feature line plots
            fg, axx = plt.subplots()          
          
            # each point individually
            maxys = []
            for pt in pts:
                axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                         r[feat][pt],color=r['cols'][pt], linewidth=0.5)
                maxys.append(np.max(r[feat][pt]))         
                         
                
            #square mean
            axx.plot(T_BIN * np.arange(len(r[feat][pt])),
                     np.mean(r[feat][pts],axis=0),
                color='k', linewidth=2)    

            axx.set_title(f'{s} \n {len(pts)} points in square')
            axx.set_xlabel('time [sec]')
            axx.set_ylabel(feat)
            
            # plot vertical boundaries for windows
            h = 0
            for i in r['len']:
            
                xv = r['len'][i] + h
                axx.axvline(T_BIN * xv, linestyle='--',
                            color='grey')
                            
                axx.text(T_BIN * xv, 0.8 * np.max(maxys), 
                         i, rotation=90, 
                         fontsize=12, color='k')
            
                h += r['len'][i]



def clus_freqs(foc='cluster', mapping='kmeans', algo_data='concat_z', nmin=50, nclus=13, vers='concat', nd=2):

    '''
    For each k-means cluster, show an Allen region bar plot of frequencies,
    or vice versa
    foc: focus, either kmeans or Allen 
    '''
    
    r_a = regional_group('Beryl', algo_data, vers=vers, nclus=nclus)    
    r_k = regional_group(mapping, algo_data, vers=vers, nclus=nclus)

    if foc == 'cluster':
    
        # show frequency of regions for all clusters
        cluss = sorted(Counter(r_k['acs']))
        fig, axs = plt.subplots(nrows = len(cluss), ncols = 1,
                               figsize=(18.79,  15),
                               sharex=True, sharey=False)
        
        fig.canvas.manager.set_window_title(
            f'Frequency of Beryl region label per'
            f' kmeans cluster ({nclus}); vers ={vers}')                      
                               
        cols_dict = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
        
        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_:
                reg_ord.append(reg)        
        
        k = 0                       
        for clus in cluss:                       
            counts = Counter(r_a['acs'][r_k['acs'] == clus])
            reg_order = {reg: 0 for reg in reg_ord}
            for reg in reg_order:
                if reg in counts:
                    reg_order[reg] = counts[reg] 
                    
            # Preparing data for plotting
            labels = list(reg_order.keys())
            values = list(reg_order.values())        
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(f'clus {clus}')
            axs[k].set_xticklabels(labels, rotation=90, 
                                   fontsize=6)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
        
        fig.tight_layout()        
        fig.subplots_adjust(top=0.951,
                            bottom=0.059,
                            left=0.037,
                            right=0.992,
                            hspace=0.225,
                            wspace=0.2)       

    else:

        # show frequency of clusters for all regions

        # order regions by canonical list 
        p = (Path(iblatlas.__file__).parent / 'beryl.npy')
        regs_can = br.id2acronym(np.load(p), mapping='Beryl')
        regs_ = Counter(r_a['acs'])
        reg_ord = []
        for reg in regs_can:
            if reg in regs_ and regs_[reg] >= nmin:
                reg_ord.append(reg)        

        print(len(reg_ord), f'regions with at least {nmin} cells')
        ncols = int((len(reg_ord) ** 0.5) + 0.999)
        nrows = (len(reg_ord) + ncols - 1) // ncols
        
        fig, axs = plt.subplots(nrows = nrows, 
                                ncols = ncols,
                                figsize=(18.79,  15),
                                sharex=True)
        
        axs = axs.flatten()
                               
        cols_dict = dict(list(Counter(zip(r_k['acs'],
                    [tuple(color) for color in r_k['cols']]))))
                    
        cols_dictr = dict(list(Counter(zip(r_a['acs'],
                                          r_a['cols']))))
        
        cluss = sorted(list(Counter(r_k['acs'])))
        
        k = 0                         
        weights = []
        for reg in reg_ord:                       
            counts = Counter(r_k['acs'][r_a['acs'] == reg])
            clus_order = {clus: 0 for clus in cluss}
            for clus in clus_order:
                if clus in counts:
                    clus_order[clus] = counts[clus] 
                    
            # Preparing data for plotting
            labels = list(clus_order.keys())
            values = list(clus_order.values())
            #weights.append([x / sum(values) for x in values])
            weights.append(values)
            colors = [cols_dict[label] for label in labels]                
                               
            # Creating the bar chart
            bars = axs[k].bar(labels, values, color=colors)
            axs[k].set_ylabel(reg, color=cols_dictr[reg])
            axs[k].set_xticks(labels)
            axs[k].set_xticklabels(labels, fontsize=8)
            
            for ticklabel, bar in zip(axs[k].get_xticklabels(), bars):
                ticklabel.set_color(bar.get_facecolor())        

            axs[k].set_xlim(-0.5, len(labels)-0.5)

            k += 1
            
        fig.canvas.manager.set_window_title(
            f'Frequency of kmeans cluster ({nclus}) per'
            f' Beryl region label per; vers = {vers}')
                     
        fig.tight_layout()
        
        centers = r_k['centers']
        plot_wasserstein_matrix(weights, centers, reg_ord, cols_dictr, vers=vers, nclus=nclus, nd=nd)
        get_difference_from_flat_dist(weights, centers, reg_ord, cols_dictr, vers=vers, nclus=nclus, nd=nd)

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'{foc}_{algo_data}_{nclus}_{vers}.png')) 
    
    #if foc=='reg':
    #    return weights, centers, reg_ord

    
def plot_wasserstein_matrix(weights, centers, reg_ord, cols_dictr, vers='concat', nclus=7, nd=2):
    # Calculate and plot wasserstein matrix of k-means clusters distributions over regions
    
    u = np.linspace(0, nclus-1, nclus)
    wass = np.zeros([len(weights),len(weights)])
    if nd==1:
        for i in range(len(weights)):
            for j in range(i):
                wass[i][j] = wasserstein_distance_nd(u,u,weights[i],weights[j])
                wass[j][i] = wass[i][j]
    elif nd>1:
        for i in range(len(weights)):
            for j in range(i):
                wass[i][j] = wasserstein_distance_nd(centers,centers,weights[i],weights[j])
                wass[j][i] = wass[i][j]
    else:
        return('what is nd')
            
    fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)
    ims = ax0.imshow(wass, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(reg_ord)), reg_ord, rotation=90, fontsize=4)
    ax0.set_yticks(np.arange(len(reg_ord)), reg_ord, fontsize=4)       
                   
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.yaxis.get_ticklabels())]
    
    ax0.set_title(f'{vers}')
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither', ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
    

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.png'), dpi=200)
    
    save_wass = {}
    save_wass['res'] = wass
    save_wass['regs'] = reg_ord
    np.save(Path(pth_dmn,f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), save_wass, allow_pickle=True)
    
    # report if any cluster(s) overrepresented in a region
    for i in range(len(weights)):
        if max(weights[i])<20:
            continue
        if sum([2*x <= max(weights[i]) for x in weights[i]]) > nclus/2:
            print(reg_ord[i], 'overrep cluster', weights[i].index(max(weights[i])))
            if sum([2*x <= np.sort(weights[i])[::-1][1] for x in weights[i]]) > nclus/2:
                print(reg_ord[i], 'overrep cluster', weights[i].index(np.sort(weights[i])[::-1][1]))



def plot_wasserstein_matrix_from_data(vers='concat', nd=2, nclus=7):
    d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), allow_pickle=True).flat[0]
    wass = d['res']
    reg_ord = d['regs']
    r_a = regional_group('Beryl', 'umap_z', vers=vers, nclus=nclus)
    cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
    
    fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)
    ims = ax0.imshow(wass, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(reg_ord)), reg_ord, rotation=90, fontsize=4)
    ax0.set_yticks(np.arange(len(reg_ord)), reg_ord, fontsize=4)       
                   
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([cols_dictr[reg] for reg in reg_ord],
        ax0.yaxis.get_ticklabels())]
    
    ax0.set_title(f'{vers}')
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither', ticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
    

    fig.savefig(Path(pth_dmn.parent, 'figs',
                     f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.png'), dpi=200)



def get_difference_from_flat_dist(weights, centers, reg_ord, cols_dictr=None, vers='concat', nclus=13, nd=2):
    '''
    Calculate wasserstein distance between the k-means clusters distributions and a flat distribution
    centers: the centers of the k clusters
    weights: count for each of the k clusters in each region
    '''
    
    wass = np.zeros(len(weights))
    flat_dist = np.ones(nclus)
    if nd==1:
        u = np.linspace(0, nclus-1, nclus)
        for i in range(len(weights)):
            wass[i] = wasserstein_distance_nd(u,u,weights[i],flat_dist)
    elif nd>1:
        for i in range(len(weights)):
            wass[i]= wasserstein_distance_nd(centers,centers,weights[i],flat_dist)
    else:
        return('what is nd')
        
    save_wass = {}
    save_wass['res'] = wass
    save_wass['regs'] = reg_ord
    np.save(Path(pth_dmn,f'wasserstein_fromflatdist_{nclus}_{vers}_nd{nd}.npy'), save_wass, allow_pickle=True)

    # plot dist in rising order for all regions
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,7), label=f'{vers}_{nclus}', dpi=150)
    ax[0].plot(np.sort(wass))
    order = np.argsort(wass)
    ax[0].set_xticks(np.arange(len(reg_ord)), np.array(reg_ord)[order], rotation=90, fontsize=4)
    ax[0].set_title(f'{vers}')
    ax[0].set_ylabel('wass_d from flat dist')

    if cols_dictr==None:
        r_a = regional_group('Beryl', 'umap_z', vers='concat', nclus=13)
        cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))
        
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in np.array(reg_ord)[order]],
        ax[0].xaxis.get_ticklabels())]

    # plot dist w/ cortical hierarchy list
    area_list = np.loadtxt(Path(pth_dmn,'area_list.csv'), dtype=str)
    plot_list = set(area_list) & set(reg_ord)
    hierarchy = [list(area_list).index(reg) for reg in plot_list]
    order = [reg_ord.index(reg) for reg in plot_list]
    color_list=[cols_dictr[reg] for reg in plot_list]
    ax[1].scatter(hierarchy, wass[order], color=color_list)
    for i, txt in enumerate(plot_list):
        ax[1].annotate(txt, (hierarchy[i], wass[order][i]))

    spearman_corr, spearman_p = spearmanr(hierarchy, wass[order])
    slope, intercept, r_value, p_value, std_err = linregress(hierarchy, wass[order])
    x=np.sort(hierarchy)
    line_fit = slope * x + intercept
    ax[1].plot(x, line_fit, color="red", label=f"Linear fit: y = {slope:.2f}x + {intercept:.2f}")
    ax[1].legend()
    ax[1].set_ylabel('wass_d from flat dist')
    ax[1].set_xlabel('position in hierarchy')
    ax[1].set_title(f'{vers}, spearman R: {spearman_corr:.2f}, p_val: {spearman_p:.4f}')
    
    fig.tight_layout
    fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_{nclus}_correlate_hierarchy.pdf'), dpi=150)



def plot_connectivity_matrix(metric='umap_z', mapping='Beryl', nclus=7, nd=2, k=2, resl=1.01,
                             vers='concat', ax0=None, clustering='hierarchy', rerun=False):

    '''
    all-to-all matrix for some measures
    '''


    if metric == 'cartesian':
        d = get_centroids(dist_=True)
    elif metric == 'pw':
        d = get_pw_dist(mapping=mapping, vers=vers)
    elif metric == 'wass':
        d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                    allow_pickle=True).flat[0]
    elif vers == 'quie-rest-diff':
        d = get_quiescence_resting_diff(metric=metric)
    else:     
        d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)
                
    res = d['res']
    regs = d['regs']
    
    _,pal = get_allen_info()
    
    alone = False
    if not ax0:
        alone=True
        if clustering=='hierarchy':
            fig, (ax_dendro, ax0) = plt.subplots(1, 2, 
                figsize=(8, 6), 
                gridspec_kw={'width_ratios': [1, 5]})
        else:
            fig, ax0 = plt.subplots(figsize=(6, 6), dpi=200)

        
    if clustering=='ari':
        rs = get_ari()
    
        ints = []
        for reg in rs:
            if reg in regs:
                ints.append(reg)
        
        rems = [reg for reg in regs if reg not in ints] 
        print(list(ints)[0], rems[0])
        node_order = list(ints) + rems
        
        ordered_indices = [list(regs).index(reg) for reg in node_order]
        regs_c = np.array(regs)[ordered_indices]
        regs_r = regs_c
        res = res[:, ordered_indices][ordered_indices, :]

    elif clustering=='dmn':
        dmn_idx = [list(regs).index(reg) for reg in dmn_regs]
        cortical_list = np.concatenate(list(cortical_regions.values()))
        cortical_list = set(cortical_list) & set(regs)
        ndmn_cortical_idx = [list(regs).index(reg) for reg in cortical_list
                    if reg not in dmn_regs]
        ndmn_idx = [list(regs).index(reg) for reg in regs
                    if reg not in cortical_list]
        ordered_indices = dmn_idx + ndmn_cortical_idx + ndmn_idx
        regs_c = np.array(regs)[ordered_indices]
        regs_r = regs_c
        res = res[:, ordered_indices][ordered_indices, :]
        

    else: 
        res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
            res, regs, k=k, metric=metric, clustering=clustering, resl=resl)
        
    
    ims = ax0.imshow(res, origin='lower', interpolation=None)
    ax0.set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=5)
    ax0.set_yticks(np.arange(len(regs_r)), regs_r, fontsize=5)       
                   
    [t.set_color(i) for (i,t) in
        zip([pal[reg] for reg in regs_c],
        ax0.xaxis.get_ticklabels())] 
         
    [t.set_color(i) for (i,t) in    
        zip([pal[reg] for reg in regs_r],
        ax0.yaxis.get_ticklabels())]
    
    if metric[-1] == 'e':
        vers = '30 ephysAtlas'
        
    ax0.set_title(f'{metric}, {vers}')
    #ax0.set_ylabel(mapping)
    cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither')#, ticks=[0, 0.5, 1]
    #cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    if clustering=='hierarchy':
        # plot dendrogram
        with plt.rc_context({'lines.linewidth': 0.5}):
            hierarchy.dendrogram(linkage_matrix, ax=ax_dendro, 
                orientation='left', labels=regs)

            
        ax_dendro.set_axis_off()
    
#    ax_dendro.set_yticklabels(regs)
#    [ax_dendro.spines[s].set_visible(False) for s in
#        ['left', 'right', 'top', 'bottom']]
#    ax_dendro.get_xaxis().set_visible(False)
#    [t.set_color(i) for (i,t) in    
#        zip([pal[reg] for reg in regs],
#        ax_dendro.yaxis.get_ticklabels())]    
    
    plt.subplots_adjust(wspace=0.05)
    fig.tight_layout()
    if clustering in ['spectralco', 'spectralbi', 'kmeans', 'kcenters', 'birch']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                         f'connectivity_matrix_{metric}_{clustering}_{vers}_k{k}.pdf'), 
                    dpi=200)
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}_k{k}.npy'),
        #       cluster_info, allow_pickle=True)
    elif clustering in ['louvain', 'leiden']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                         f'connectivity_matrix_{metric}_{clustering}_{vers}_{resl}.pdf'), 
                    dpi=200)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                         f'connectivity_matrix_{metric}_{clustering}_{vers}.pdf'), 
                    dpi=200)
        
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}.npy'),
        #       cluster_info, allow_pickle=True)
    
    #if alone:
    #    fig.tight_layout()

    #fig0.suptitle(f'{algo}, {mapping}')
    #else:
    if not alone:
        return cluster_info


def plot_all_connectivity_matrices(mapping='Beryl', algo='umap_z', nclus=7, nd=2, k_range=[2,3,4,5,6,7],
                                   resl_range=[1,1.01,1.02,1.03,1.04,1.05,1.06,1.07],
                                   vers='concat', ax0=None, rerun=False):

    '''
    all-to-all matrix for all clustering measures
    '''

    fig, axs = plt.subplots(nrows=7, ncols=len(k_range),
                            figsize=(10,10), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    listr = {}
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch',
                       'spectralco', 'spectralbi', 'kmeans']:
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = algo
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res = d['res']
        regs = d['regs']

        if n<2:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res=res, regs=regs, k=None, metric=metric, clustering=clustering)
            
            ims = axs[n].imshow(res, origin='lower', interpolation=None)
            axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
            axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
            [t.set_color(i) for (i,t) in
                zip([pal[reg] for reg in regs_c],
                axs[n].xaxis.get_ticklabels())] 
         
            [t.set_color(i) for (i,t) in    
                zip([pal[reg] for reg in regs_r],
                axs[n].yaxis.get_ticklabels())]

            if n==0:
                axs[n].set_title(f'wass_{clustering}', fontsize=10)
            else:
                axs[n].set_title(f'{clustering}', fontsize=10)
            cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                        extend='neither')#, ticks=[0, 0.5, 1]
            cbar.ax.tick_params(labelsize=5)
            
            n=n+1
            
        elif clustering in ['louvain', 'leiden']:
            for resl in resl_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res=res, regs=regs, resl=resl, metric=metric, clustering=clustering)
                
                ims = axs[n].imshow(res, origin='lower', interpolation=None)
                axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
                axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
                [t.set_color(i) for (i,t) in
                    zip([pal[reg] for reg in regs_c],
                    axs[n].xaxis.get_ticklabels())] 
         
                [t.set_color(i) for (i,t) in    
                    zip([pal[reg] for reg in regs_r],
                    axs[n].yaxis.get_ticklabels())]
            
                axs[n].set_title(f'{clustering}, resl{resl}', fontsize=10)
                cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                            extend='neither')#, ticks=[0, 0.5, 1]
                cbar.ax.tick_params(labelsize=5)
                                
                n=n+1

        else:
            for k in k_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res=res, regs=regs, k=k, metric=metric, clustering=clustering)
                
                ims = axs[n].imshow(res, origin='lower', interpolation=None)
                axs[n].set_xticks(np.arange(len(regs_c)), regs_c,
                   rotation=90, fontsize=1)
                axs[n].set_yticks(np.arange(len(regs_r)), regs_r, fontsize=1)       
                   
                [t.set_color(i) for (i,t) in
                    zip([pal[reg] for reg in regs_c],
                    axs[n].xaxis.get_ticklabels())] 
         
                [t.set_color(i) for (i,t) in    
                    zip([pal[reg] for reg in regs_r],
                    axs[n].yaxis.get_ticklabels())]
            
                axs[n].set_title(f'{clustering}, k{k}', fontsize=10)
                cbar = plt.colorbar(ims,fraction=0.046, pad=0.04, 
                            extend='neither')#, ticks=[0, 0.5, 1]
                cbar.ax.tick_params(labelsize=5)
                                
                n=n+1
                
    
    fig.suptitle(vers)
    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'all_connectivity_matrices_{vers}_{algo}.pdf'), 
                dpi=400)



def plot_avg_peth_from_clustering(vers, clustering, k=None, resl=None):
    d = get_reg_dist(algo='umap_z', vers=vers)
    _, regs_r, _, info0 = clustering_on_connectivity_matrix(
        d['res'], d['regs'], k=k, clustering=clustering, resl=resl)
    info0 = np.sort(info0) # sort cluster labels for regions
    r = regional_group(mapping='Beryl', algo='umap_z', vers=vers, nclus=7) # get peth data

    feat = 'concat_z'
    fg, axx = plt.subplots(nrows=len(np.unique(info0)),
                           sharex=True, sharey=False,
                           figsize=(6,6))
                        
    kk = 0
    for clu in np.unique(info0):
            print('cluster:', clu)
            
            #cluster mean
            listr = regs_r[info0==clu] #list of regions in a cluster
            print('regs in the cluster:', listr)
            xx = np.arange(len(r[feat][0])) /480
            yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)

            axx[kk].plot(xx, yy, linewidth=2)
                     

            
            if kk != (len(np.unique(r['acs'])) - 1):
                axx[kk].axis('off')
            else:

                axx[kk].spines['top'].set_visible(False)
                axx[kk].spines['right'].set_visible(False)
                axx[kk].spines['left'].set_visible(False)      
                axx[kk].tick_params(left=False, labelleft=False)
                
            d2 = {}
            for sec in PETH_types_dict[vers]:
                d2[sec] = r['len'][sec]
                                
            # plot vertical boundaries for windows
            h = 0
            for i in d2:
            
                xv = d2[i] + h
                axx[kk].axvline(xv/480, linestyle='--', linewidth=1,
                            color='grey')
                
                if  kk == 0:            
                    axx[kk].text(xv/480 - d2[i]/(2*480), max(yy),
                             '   '+i, rotation=90, color='k', 
                             fontsize=10, ha='center')
            
                h += d2[i] 
            kk += 1
        
    axx[kk - 1].set_xlabel('time [sec]')
    #axx.set_ylabel(feat)
    fg.tight_layout()
    fg.savefig(Path(one.cache_dir,'dmn', 'figs',
       f'{vers}_avg_peth_from_{clustering}_k{k}_resl{resl}.png'), dpi=150, bbox_inches='tight')


def plot_avg_peth_from_all_clustering(vers, nd=2, rerun=False, k_range=[2,3,4,5,6]):

    fig = plt.figure(figsize=(10,10), dpi=400)
    outer_grid = gridspec.GridSpec(4, len(k_range), wspace=0.4, hspace=0.4)
    
    n=0
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch', 
                       'spectralco', 'spectralbi', 'kmeans']:
        print(clustering)

        # load data from connectivity matrix
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_7_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = 'umap_z'
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)

        # perform clustering and plot results
        if n<5: # first five methods without a k value
            _, regs_r, _, info0 = clustering_on_connectivity_matrix(
                d['res'], d['regs'], k=None, metric=metric, clustering=clustering)
            
            info0 = np.sort(info0)
            # load peth data
            r = regional_group(mapping='Beryl', algo='umap_z', vers=vers, nclus=7)
            feat = 'concat_z'

            # plot nth subplot
            inner_grid = gridspec.GridSpecFromSubplotSpec(len(np.unique(info0)), 1, 
                            subplot_spec=outer_grid[n])
            outer_ax = fig.add_subplot(outer_grid[n])
            outer_ax.set_title(clustering)  # Title for outer subplot
            outer_ax.set_frame_on(False)  # Hide the frame of the outer axis
            outer_ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

                        
            kk = 0
            for clu in np.unique(info0):
                print('cluster:', clu)
            
                #cluster mean
                listr = regs_r[info0==clu] #list of regions in a cluster
                print('regs in the cluster:', listr)
                xx = np.arange(len(r[feat][0])) /480
                yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)

                axx = fig.add_subplot(inner_grid[kk])
                axx.plot(xx, yy, linewidth=2)
                
                if kk != (len(np.unique(r['acs'])) - 1):
                    axx.axis('off')
                else:

                    axx.spines['top'].set_visible(False)
                    axx.spines['right'].set_visible(False)
                    axx.spines['left'].set_visible(False)      
                    axx.tick_params(left=False, labelleft=False)
                
                d2 = {}
                for sec in PETH_types_dict[vers]:
                    d2[sec] = r['len'][sec]
                                
                # plot vertical boundaries for windows
                h = 0
                for i in d2:
            
                    xv = d2[i] + h
                    axx.axvline(xv/480, linestyle='--', linewidth=1,
                                    color='grey')
                
                    #if  kk == 0:            
                    #    axx.text(xv/480 - d2[i]/(2*480), max(yy),
                    #         '   '+i, rotation=90, color='k', 
                    #         fontsize=10, ha='center')
            
                    h += d2[i] 
                kk += 1
        
            axx.set_xlabel('time [sec]')

            n = n+1                     
            
        else: # clustering methods with specified k
            for k in k_range:
                print('k =', k)
                _, regs_r, _, info0 = clustering_on_connectivity_matrix(
                    d['res'], d['regs'], k=k, metric=metric, clustering=clustering)
                
                # plot nth subplot
                inner_grid = gridspec.GridSpecFromSubplotSpec(len(np.unique(info0)), 1, 
                            subplot_spec=outer_grid[n])
                outer_ax = fig.add_subplot(outer_grid[n])
                outer_ax.set_title(f'{clustering}_{k}')  # Title for outer subplot
                outer_ax.set_frame_on(False)  # Hide the frame of the outer axis
                outer_ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        
                kk = 0
                for clu in np.unique(info0):
                    print('cluster:', clu)
            
                    #cluster mean
                    listr = regs_r[info0==clu] #list of regions in a cluster
                    print('regs in the cluster:', listr)
                    xx = np.arange(len(r[feat][0])) /480
                    yy = np.mean(r[feat][np.where(np.isin(r['acs'], listr))], axis=0)

                    axx = fig.add_subplot(inner_grid[kk])
                    axx.plot(xx, yy, linewidth=2)
                
                    if kk != (len(np.unique(r['acs'])) - 1):
                        axx.axis('off')
                    else:

                        axx.spines['top'].set_visible(False)
                        axx.spines['right'].set_visible(False)
                        axx.spines['left'].set_visible(False)      
                        axx.tick_params(left=False, labelleft=False)
                
                    d2 = {}
                    for sec in PETH_types_dict[vers]:
                        d2[sec] = r['len'][sec]
                                
                    # plot vertical boundaries for windows
                    h = 0
                    for i in d2:
            
                        xv = d2[i] + h
                        axx.axvline(xv/480, linestyle='--', linewidth=1,
                                    color='grey')
                
                        #if  kk == 0:            
                        #    axx.text(xv/480 - d2[i]/(2*480), max(yy),
                        #         '   '+i, rotation=90, color='k', 
                        #         fontsize=10, ha='center')
            
                        h += d2[i] 
                    kk += 1

                n = n+1
            axx.set_xlabel('time [sec]')

    fig.suptitle(vers, fontsize=20)
    fig.tight_layout()    
    fig.savefig(Path(one.cache_dir,'dmn', 'figs',
        f'{vers}_avg_peth_from_all_clustering.png'), dpi=400, bbox_inches='tight')


def plot_connectivity_network_style(vers, clustering, metric='umap_z', 
                                    k=2, resl=1.01, layout='shell', coloring='Beryl',
                                    diff='shifted', threshold=0.8, edge_display=0.1):

    # get data from clustering
    if vers == 'quie-rest-diff':
        d = get_quiescence_resting_diff(metric=metric, diff=diff)
    else:
        d = get_reg_dist(algo=metric, vers=vers)
    res = d['res']
    regs = d['regs']    
    res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
            res, regs, k=k, metric=metric, clustering=clustering, resl=resl)

    # Add graph nodes and edges with weights (connectivity strengths)
    G = nx.Graph()
    G.add_nodes_from(regs_r)
    for i in range(len(regs_r)):
        for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
            if res[i, j]!=0:  # Only add edges for nonzero connectivity
                G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])

    # Define a layout for the nodes
    if layout=='spring':
        pos = nx.spring_layout(G)
    elif layout=='shell':
        pos = nx.shell_layout(G)
    elif layout=='kamada':
        pos = nx.kamada_kawai_layout(G)
    elif layout=='manual': #manually put nodes in the same cluster near each other
        pos = nx.kamada_kawai_layout(G)
        clusters = np.sort(cluster_info)
        for cluster in set(clusters):
            cluster_nodes = regs_r[clusters==cluster]
            cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
            for node in cluster_nodes:
                pos[node] = 0.05 * pos[node] + 0.95 * cluster_center
    
    # Draw nodes
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 11), dpi=200)
    if coloring=='dmn':
        cmap0 = cm.get_cmap('tab10')
        cols = [cmap0(int(reg in dmn_regs)) for reg in G.nodes()]
        transparency = 1
    elif coloring=='cortical':
        pal={}
        for reg in G.nodes():
            pal[reg] = 'white'
            for cortical in cortical_regions.keys():
                        if reg in cortical_regions[cortical]:
                            pal[reg] = cortical_colors[cortical]
        cols = [pal[reg] for reg in G.nodes()]
        transparency = [(int(color!='white')+1)/2 for color in cols]
    else:
        _, pal = get_allen_info()
        cols = [pal[reg] for reg in G.nodes()]
        transparency = 1
    cmap = cm.get_cmap('tab20')
    node_colors = [cmap(i) for i in np.sort(cluster_info)]
    nx.draw_networkx_nodes(G, pos, ax=axs[0], node_color=cols, alpha=transparency, node_size=40)
    nx.draw_networkx_nodes(G, pos, ax=axs[1], node_color=node_colors, node_size=40)

    # Draw edges with varying thickness based on connectivity strength
    edges = G.edges(data=True)
    strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[0],
        width=[d['weight']*edge_display for (u, v, d) in 
               G.edges(data=True) if d['weight']> threshold])  # Adjust thickness based on weight
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[1],
        width=[d['weight']*edge_display for (u, v, d) in 
               G.edges(data=True) if d['weight']> threshold])  # Adjust thickness based on weight

    # Draw labels for the nodes
    nx.draw_networkx_labels(G, pos, ax=axs[0], font_size=4)
    nx.draw_networkx_labels(G, pos, ax=axs[1], font_size=4)

    axs[0].set_title(f'colored by {coloring}', size=20)
    axs[1].set_title(f'colored by {clustering} clustering', size=20)
    fig.suptitle(f'{vers}, {clustering}', fontsize=40)

    axs[0].axis('off')
    axs[1].axis('off')
    fig.tight_layout()
    if clustering in ['spectralco', 'spectralbi', 'kmeans', 'kcenters', 'birch']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                         f'network_{metric}_{clustering}_{vers}_k{k}_thr{threshold}_{layout}.pdf'), 
                    dpi=200)
        #np.save(Path(pth_dmn.parent, 'res', 
        #                 f'cluster_info_{metric}_{clustering}_{vers}_k{k}.npy'),
        #       cluster_info, allow_pickle=True)
    elif clustering in ['louvain', 'leiden']:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'network_{metric}_{clustering}_{vers}_{resl}_thr{threshold}_{layout}_{coloring}.pdf'), 
                    dpi=200)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'network_{metric}_{clustering}_{vers}_thr{threshold}_{layout}_{coloring}.pdf'), 
                    dpi=200)

    plt.show()



def plot_all_connectivity_networks(mapping='Beryl', algo='umap_z', nclus=7, nd=2, edge_display=0.1, k_range=[2,3,4,5,6,7],
                                   resl_range=[1,1.01,1.02,1.03,1.04,1.05,1.06,1.07], threshold=0.9,
                                   vers='concat', layout='shell', coloring='Beryl', rerun=False):

    '''
    network style connectivity plots for all clustering measures & parameters
    '''

    fig, axs = plt.subplots(nrows=7, ncols=len(k_range),
                            figsize=(10,12), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    for clustering in ['hierarchy', 'hierarchy', 'louvain', 'leiden', 'birch',
                       'spectralco', 'spectralbi', 'kmeans']:
        if n==0:
            metric='wass'
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            metric = algo
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res0 = d['res']
        regs = d['regs']

        
        if n<2:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res0, regs, k=None, metric=metric, clustering=clustering)
            
            # Add graph nodes and edges with weights (connectivity strengths)
            G = nx.Graph()
            G.add_nodes_from(regs_r)
            for i in range(len(regs_r)):
                for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                    if res[i, j]!=0:  # Only add edges for nonzero connectivity
                        G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                        
            # Define a layout for the nodes
            if layout=='spring':
                pos = nx.spring_layout(G)
            elif layout=='shell':
                pos = nx.shell_layout(G)
            elif layout=='kamada':
                pos = nx.kamada_kawai_layout(G)
            elif layout=='manual': #manually put nodes in the same cluster near each other
                pos = nx.kamada_kawai_layout(G)
                clusters = np.sort(cluster_info)
                for cluster in set(clusters):
                    cluster_nodes = regs_r[clusters==cluster]
                    cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                    for node in cluster_nodes:
                        pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
            # Plot graph
            if coloring=='Beryl':
                _, pal = get_allen_info()
                cols = [pal[reg] for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
            else:
                cmap = cm.get_cmap('tab20')
                node_colors = [cmap(i) for i in np.sort(cluster_info)]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)

            edges = G.edges(data=True)
            strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
            # Adjust thickness based on weight
            nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            if n==0:
                axs[n].set_title(f'wass_{clustering}', fontsize=10)
            else:
                axs[n].set_title(f'{clustering}', fontsize=10)

            #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
            axs[n].axis('off')
            n=n+1

        
        elif clustering in ['louvain', 'leiden']:
            for resl in resl_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res0, regs, resl=resl, metric=metric, clustering=clustering)
                
                # Add graph nodes and edges with weights (connectivity strengths)
                G = nx.Graph()
                G.add_nodes_from(regs_r)
                for i in range(len(regs_r)):
                    for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                        if res[i, j]!=0:  # Only add edges for nonzero connectivity
                            G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                            
                # Define a layout for the nodes
                if layout=='spring':
                    pos = nx.spring_layout(G)
                elif layout=='shell':
                    pos = nx.shell_layout(G)
                elif layout=='kamada':
                    pos = nx.kamada_kawai_layout(G)
                elif layout=='manual': #manually put nodes in the same cluster near each other
                    pos = nx.kamada_kawai_layout(G)
                    clusters = np.sort(cluster_info)
                    for cluster in set(clusters):
                        cluster_nodes = regs_r[clusters==cluster]
                        cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                        for node in cluster_nodes:
                            pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
                # Plot graph
                if coloring=='Beryl':
                    _, pal = get_allen_info()
                    cols = [pal[reg] for reg in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
                else:
                    cmap = cm.get_cmap('tab20')
                    node_colors = [cmap(i) for i in np.sort(cluster_info)]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
                edges = G.edges(data=True)
                strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
            
                # Adjust thickness based on weight
                nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            
                axs[n].set_title(f'{clustering}, resl{resl}', fontsize=10)
                #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
                axs[n].axis('off')
                n=n+1

        else:
            for k in k_range:
                res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                    res0, regs, k=k, metric=metric, clustering=clustering)
                
                # Add graph nodes and edges with weights (connectivity strengths)
                G = nx.Graph()
                G.add_nodes_from(regs_r)
                for i in range(len(regs_r)):
                    for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                        if res[i, j]!=0:  # Only add edges for nonzero connectivity
                            G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                            
                # Define a layout for the nodes
                if layout=='spring':
                    pos = nx.spring_layout(G)
                elif layout=='shell':
                    pos = nx.shell_layout(G)
                elif layout=='kamada':
                    pos = nx.kamada_kawai_layout(G)
                elif layout=='manual': #manually put nodes in the same cluster near each other
                    pos = nx.kamada_kawai_layout(G)
                    clusters = np.sort(cluster_info)
                    for cluster in set(clusters):
                        cluster_nodes = regs_r[clusters==cluster]
                        cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                        for node in cluster_nodes:
                            pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
                # Plot graph
                if coloring=='Beryl':
                    _, pal = get_allen_info()
                    cols = [pal[reg] for reg in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
                else:
                    cmap = cm.get_cmap('tab20')
                    node_colors = [cmap(i) for i in np.sort(cluster_info)]
                    nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
                edges = G.edges(data=True)
                strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
                
                # Adjust thickness based on weight
                nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
            
            
                axs[n].set_title(f'{clustering}, k{k}', fontsize=10)
                #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
                axs[n].axis('off')
                n=n+1
                
    
    fig.suptitle(vers)
    fig.tight_layout()
    fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'all_connectivity_networks_{coloring}_{vers}_{algo}_{threshold}_{layout}.pdf'), 
                dpi=400)



def plot_connec_networks_over_time(clustering, layout='manual', nclus=13, nd=2, edge_display=0.1,
                                   k=None, resl=None, threshold=0.9, top_n=100, metric='umap_z', 
                                   coloring='Beryl', rerun=False):

    '''
    network style connectivity plots for all clustering measures & parameters
    '''

    fig, axs = plt.subplots(nrows=3, ncols=3,
                            figsize=(9,10), dpi=400)
    axs = axs.flatten()
    _,pal = get_allen_info()

    n = 0
    for vers in ['concat', 'resting', 'quiescence', 'pre-stim-prior', 
                 'stim_surp_con', 'stim_surp_incon', 'motor_init', 'fback1', 'fback0']:
        if metric=='wass':
            d = np.load(Path(pth_dmn, f'wasserstein_matrix_{nclus}_{vers}_nd{nd}.npy'), 
                        allow_pickle=True).flat[0]
        else:
            d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)            
            
        res0 = d['res']
        regs = d['regs']

        if layout=='shell':
            # order regions by canonical list 
            p = (Path(iblatlas.__file__).parent / 'beryl.npy')
            regs_can = br.id2acronym(np.load(p), mapping='Beryl')
            regs_r,reg_ord = [],[]
            for reg in regs_can:
                if reg in regs:
                    regs_r.append(reg)
                    reg_ord.append(np.where(regs==reg)[0][0])

            res=res0[reg_ord]
            #coloring='Beryl'
        else:
            res, regs_r, regs_c, cluster_info = clustering_on_connectivity_matrix(
                res0, regs, k=k, resl=resl, metric=metric, clustering=clustering)
            
        # Add graph nodes and edges with weights (connectivity strengths)
        G = nx.Graph()
        G.add_nodes_from(regs_r)
        for i in range(len(regs_r)):
            for j in range(i+1, len(regs_r)):  # Use only upper triangle for undirected graph
                if res[i, j]!=0:  # Only add edges for nonzero connectivity
                        G.add_edge(regs_r[i], regs_r[j], weight=res[i, j])
                        
        # Define a layout for the nodes
        if layout=='spring':
                pos = nx.spring_layout(G)
        elif layout=='shell':
                pos = nx.shell_layout(G)
        elif layout=='kamada':
                pos = nx.kamada_kawai_layout(G)
        elif layout=='manual': #manually put nodes in the same cluster near each other
                pos = nx.kamada_kawai_layout(G)
                clusters = np.sort(cluster_info)
                for cluster in set(clusters):
                    cluster_nodes = regs_r[clusters==cluster]
                    cluster_center = np.mean([pos[node] for node in cluster_nodes], axis=0)
                    for node in cluster_nodes:
                        pos[node] = 0.1 * pos[node] + 0.9 * cluster_center
                
        # Plot graph
        if coloring=='Beryl':
                _, pal = get_allen_info()
                cols = [pal[reg] for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=cols, node_size=2)
        elif coloring=='dmn':
                cmap = cm.get_cmap('tab10')
                node_colors = [cmap(int(reg in dmn_regs)) for reg in G.nodes()]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)
        elif coloring=='cortical':
                pal={}
                for reg in G.nodes():
                    pal[reg] = 'white'
                    for cortical in cortical_regions.keys():
                        if reg in cortical_regions[cortical]:
                            pal[reg] = cortical_colors[cortical]
                node_colors = [pal[reg] for reg in G.nodes()]
                transparency = [(int(color!='white')+1)/2 for color in node_colors]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, 
                                       alpha=transparency, node_size=4)
        else:
                cmap = cm.get_cmap('tab20')
                node_colors = [cmap(i) for i in np.sort(cluster_info)]
                nx.draw_networkx_nodes(G, pos, ax=axs[n], node_color=node_colors, node_size=4)

        edges = G.edges(data=True)
        strong_edges = [(i, j) for i, j, w in G.edges(data=True) if w['weight'] > threshold]
        # Adjust thickness based on weight
        nx.draw_networkx_edges(G, pos, edgelist=strong_edges, ax=axs[n], 
                                   width=[d['weight']*edge_display for (u, v, d) in 
                                          G.edges(data=True) if d['weight']> threshold]) 
        axs[n].set_title(f'{vers}', fontsize=10)

        #nx.draw_networkx_labels(G, pos, ax=axs[n], font_size=1)
        axs[n].axis('off')
        n=n+1
                
    if layout=='shell':
        fig.suptitle(f'canonical ordering')
    else:
        fig.suptitle(f'{metric}_{clustering}_k{k}_resl{resl}')
    fig.tight_layout()
    if layout=='shell':
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'conn_networks_over_time_{metric}_{coloring}_{threshold}_{layout}.pdf'), 
                    dpi=400)
    else:
        fig.savefig(Path(pth_dmn.parent, 'figs', 
                    f'conn_networks_over_time_{metric}_{coloring}_{clustering}_k{k}resl{resl}_{threshold}_{layout}.pdf'), 
                    dpi=400)



def plot_avg_corr_with_dmn_regions(vers, metric='umap_z', rerun=False, cols_dictr=None,
                                   only_cortical=False):

    if cols_dictr==None:
        r_a = regional_group('Beryl', 'umap_z', vers='concat', nclus=13)
        cols_dictr = dict(list(Counter(zip(r_a['acs'], r_a['cols']))))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), label=f'{vers}', dpi=150)
    cmap = cm.get_cmap('tab10')

    d = get_reg_dist(algo=metric, vers=vers, rerun=rerun)
    dmn_idx = [list(d['regs']).index(reg) for reg in dmn_regs]
    if only_cortical:
        cortical_list = np.concatenate(list(cortical_regions.values()))
        cortical_list = set(cortical_list) & set(d['regs'])
        cortical_idx = [list(d['regs']).index(reg) for reg in cortical_list]
        d['res'] = d['res'][:,cortical_idx]
        ndmn_idx = [list(d['regs']).index(reg) for reg in cortical_list
                    if reg not in dmn_regs]
        d['regs'] = d['regs'][cortical_idx]
    else:
        ndmn_idx = [list(d['regs']).index(reg) for reg in d['regs']
                    if reg not in dmn_regs]
    avg_corr_dmn = np.mean(d['res'][dmn_idx,:], axis=0)
    avg_corr_ndmn = np.mean(d['res'][ndmn_idx,:], axis=0)

    order = np.argsort(avg_corr_dmn)
    colors = [cmap(int(reg in dmn_regs)) for reg in d['regs'][order]]
    ax.scatter(d['regs'][order], avg_corr_dmn[order], color=colors, s=7, label='with dmn regs')
    ax.scatter(d['regs'][order], avg_corr_ndmn[order], color=colors, s=5, marker='v', label='with non-dmn regs')
    ax.set_xticks(np.arange(len(d['regs'])), d['regs'][order], rotation=90, fontsize=4)
    [t.set_color(i) for (i,t) in
        zip([cols_dictr[reg] for reg in d['regs'][order]],
        ax.xaxis.get_ticklabels())]
    ax.set_title(f'{vers}, average correlation per region', size=10)
    ax.legend()

    fig.tight_layout
    if only_cortical:
        fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_avg_corr_dmn_cortical.pdf'), dpi=150)
    else:
        fig.savefig(Path(one.cache_dir,'dmn', 'figs', 
                     f'{vers}_avg_corr_dmn.pdf'), dpi=150)