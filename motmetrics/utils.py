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
from .distances import sen_iou_matrix, sen_norm2squared_matrix
from .preprocess import preprocessResult, preprocessResult_det

import motmetrics.io as io

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

def CLEAR_MOT_M_senseTk(gt, dt, inifile, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5, include_all = False, log = '', det = None, label=None, fmt = 'mot16'):
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
        return sen_iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return sen_norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()
    fragments = {}
    #import time
    #print('preprocess start.')
    #pst = time.time()
    if fmt=='mot16':
        # print('before', dt.count())
        dt = preprocessResult(dt, gt, inifile)
        # print('after', dt.count())
        if det is not None:
            det = preprocessResult_det(det, gt, inifile, label)
    #pen = time.time()
    #print('preprocess take ', pen - pst)
        if include_all:
            # gt = gt[gt['Confidence'] >= 0.99]
            new_gt = gt.__class__()
            for i in gt.allFr():
                for j in gt[i]:
                    if j.conf>=0.99:
                        new_gt.append_data(j)
            gt = new_gt
        else:
            # gt = gt[ (gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1) ]
            new_gt = gt.__class__()
            for i in gt.allFr():
                for j in gt[i]:
                    if j.conf>=0.99 and j.label==1:
                        new_gt.append_data(j)
            gt = new_gt
    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in 
    # case a frame is missing in detector results this will lead to FNs.
    be = min(gt.min_fr, dt.min_fr)
    en = max(gt.max_fr, dt.max_fr)
    allframeids = range(be, en+1)

    analysis = {}
    m_plus = {(d+k):0 for d in list('YN') for k in ['Match', 'Track', 'FP', 'FN']}
    m_plus['Filter'] = 0
    for fid in allframeids:
        #st = time.time()
        oids = []
        hids = []
        dists = np.empty((0,0))


        for j in gt[fid]:
            oid = j.uid
            oids.append(oid)
            # analysis['obj'][oid] = analysis['obj'].get(oid, 0) + 1
        oids = np.array(oids, dtype=np.int32)
        for j in dt[fid]:
            hid = j.uid
            hids.append(hid)
            # analysis['hyp'][hid] = analysis['hyp'].get(hid, 0) + 1
        hids = np.array(hids, dtype=np.int32)

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(gt[fid], dt[fid])
        dids = None
        if det is not None:
            dgt = {i: None for i in oids}
            dhy = {i: None for i in hids}
            dd = det[fid]
            dids = {'det_gt': dgt, 'det_hyp': dhy, 'det': dd}
            if dd is not None:
                for j in gt[fid]:
                    dgt[j.uid] = j
                for j in dt[fid]:
                    dhy[j.uid] = j
                    #print(d)
                    #print(mx_id[1])
        #print('gt', oids)
        #print('dt', hids)
        #print(dists.shape)
        acc.update(oids, hids, dists, frameid=fid, log = log, metric_plus = dids, metrics_qa = analysis)
        if dids:
            for k in m_plus:
                m_plus[k]+=dids[k]
        #en = time.time()
        #print(fid, ' time ', en - st)
    if det is not None:
        analysis['metric_plus'] = m_plus
    return acc, analysis


def CLEAR_MOT_M(gt, dt, inifile, dist='iou', distfields=['X', 'Y', 'Width', 'Height'], distth=0.5, include_all = False, log = '', det = None, label=None, fmt = 'mot16'):
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

    if io.engine_type=='senseTk':
        return CLEAR_MOT_M_senseTk(gt, dt, inifile, dist, distfields, distth, include_all, log, det, label, fmt)

    def compute_iou(a, b):
        return iou_matrix(a, b, max_iou=distth)

    def compute_euc(a, b):
        return norm2squared_matrix(a, b, max_d2=distth)

    compute_dist = compute_iou if dist.upper() == 'IOU' else compute_euc

    acc = MOTAccumulator()
    #import time
    #print('preprocess start.')
    #pst = time.time()
    if fmt=='mot16':
        dt = preprocessResult(dt, gt, inifile)
        if det is not None:
            det = preprocessResult_det(det, gt, inifile, label)
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
    analysis = {}
    m_plus = {(d+k):0 for d in list('YN') for k in ['Match', 'Track', 'FP', 'FN']}
    m_plus['Filter'] = 0
    for fid in allframeids:
        #st = time.time()
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
        dids = None
        #print(det)
        #print(dt)
        #print(gt)
        if det is not None:
            dgt = {i: None for i in oids}
            dhy = {i: None for i in hids}
            if fid in det.index:
                dd = det.loc[fid].values
            else:
                dd = []
            dids = {'det_gt': dgt, 'det_hyp': dhy, 'det': dd}
            if dd is not None:
                if fid in gt.index:
                    fgt = gt.loc[fid]
                    for oid in oids:
                        dgt[oid] = fgt.loc[oid].values
                if fid in dt.index:
                    fdt = dt.loc[fid]
                    for hid in hids:
                        dhy[hid] = fdt.loc[hid].values
                    #print(d)
                    #print(mx_id[1])

        acc.update(oids, hids, dists, frameid=fid, log = log, metric_plus = dids, metrics_qa = analysis)
        if dids:
            for k in m_plus:
                m_plus[k]+=dids[k]
        #en = time.time()
        #print(fid, ' time ', en - st)
    if det is not None:
        analysis['metric_plus'] = m_plus
    return acc, analysis
