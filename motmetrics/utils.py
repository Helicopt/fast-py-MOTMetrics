"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.

Christoph Heindl, 2017
Toka, 2018
https://github.com/cheind/py-motmetrics
TOKA EXTENDED THIS FILE.
"""

import pandas as pd
import numpy as np

from .mot import MOTAccumulator
from .distances import iou_matrix, norm2squared_matrix
from .preprocess import preprocessResult

def compare_to_groundtruth(gt, dt, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids
    
    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results
    
    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in 
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0,0))

        if fid in gt.index:
            fgt = gt.loc[fid] 
            oids = fgt.index.values

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)
        
        acc.update(oids, hids, dists, frameid=fid)
    
    return acc

def CLEAR_MOT_M(gt, dt, inifile, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5, include_all = False, vflag = '', det = None):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids
    
    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    test : pd.DataFrame
        Dataframe for detector results
    
    Kwargs
    ------
    dist : str, optional
        String identifying distance to be used. Defaults to intersection over union.
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    distth: float, optional
        Maximum tolerable distance. Pairs exceeding this threshold are marked 'do-not-pair'.
    """

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()
    #import time
    #print('preprocess start.')
    #pst = time.time()
    dt = preprocessResult(dt, gt, inifile)
    if det is not None:
        det = preprocessResult(det, gt, inifile)
    #pen = time.time()
    #print('preprocess take ', pen - pst)
    if include_all:
        gt = gt[gt['Confidence'] >= 0.99]
    else:
        gt = gt[ (gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1) ]
    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in 
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]
    analysis = {'hyp':{}, 'obj':{}}
    m_plus = {(d+k):0 for d in list('YN') for k in ['Match', 'Track', 'FP', 'FN']}
    def IoU_(xx, yy):
        x1, y1, w1, h1 = xx
        x2, y2, w2, h2 = yy
        mx1 = max(x1, x2)
        my1 = max(y1, y2)
        mx2 = min(x1+w1, x2+w2)
        my2 = min(y1+h1, y2+h2)
        ix = max(mx2 - mx1, 0.)
        iy = max(my2 - my1, 0.)
        intersec = ix * iy
        iou = intersec / (w1*h1+w2*h2-intersec)
        return iou
    IoU = lambda x, y: IoU_(x[:4], y[:4])
    def has_det_cover(d, x):
        for det in d.values:
            if IoU(det, x)>0.5:
                return True
        return False
    for fid in allframeids:
        #st = time.time()
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0,0))

        if fid in gt.index:
            fgt = gt.loc[fid] 
            oids = fgt.index.values
            for oid in oids:
                oid = int(oid)
                if oid not in analysis['obj']:
                    analysis['obj'][oid] = 0
                analysis['obj'][oid] += 1

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values
            for hid in hids:
                hid = int(hid)
                if hid not in analysis['hyp']:
                    analysis['hyp'][hid] = 0
                analysis['hyp'][hid] += 1

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[distfields].values, fdt[distfields].values)
        dids = None
        if det is not None:
            dgt = {}
            dhy = {}
            dids = {'det_gt': dgt, 'det_hyp': dhy}
            if fid in det.index:
                dd = det.loc[fid]
            else:
                dd = None
            if fid in dt.index:
                fdt = dt.loc[fid]
                for hid in hids:
                    dhy[hid] = False if dd is None else has_det_cover(dd, fdt.loc[hid].values)
            if fid in gt.index:
                fgt = gt.loc[fid]
                for oid in oids:
                    dgt[oid] = False if dd is None else has_det_cover(dd, fgt.loc[oid].values)

        acc.update(oids, hids, dists, frameid=fid, vf = vflag, metric_plus = dids)
        if dids:
            for k in m_plus:
                m_plus[k]+=dids[k]
        #en = time.time()
        #print(fid, ' time ', en - st)
    if det is not None:
        analysis['metric_plus'] = m_plus
    return acc, analysis
