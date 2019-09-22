"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking with RESULT PREPROCESS.

TOKA, 2018
ORIGIN: https://github.com/cheind/py-motmetrics
EXTENDED: <reposity>
"""

import argparse
import glob
import os
import logging
import motmetrics as mm
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import time
from tempfile import NamedTemporaryFile

def parse_args():
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in 

Milan, Anton, et al. 
"Mot16: A benchmark for multi-object tracking." 
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Seqmap for test data
    [name]
    <SEQUENCE_1>
    <SEQUENCE_2>
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('seqmap', type=str, help='Text file containing all sequences name')
    parser.add_argument('-d', '--detections', type=str, default=None, help='Text file containing detection files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--label', type=str, default=None, help='class label for drop detection results')
    parser.add_argument('--block', type=int, default=1, help='block frames n means choosing first frame as key frame for every n frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    parser.add_argument('-k', default=False, action='store_true', help='Use K-Score metrics')
    return parser.parse_args()

def compare_dataframes(gts, ts, vsflag = '', iou = 0.5, det = None, label=None):
    accs = []
    anas = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logging.info('Evaluating {}...'.format(k))
            if vsflag!='':
                fd = open(vsflag+'/'+k+'.log','w')
            else:
                fd = ''
            acc, ana = mm.utils.CLEAR_MOT_M(gts[k][0], tsacc, gts[k][1], 'iou', distth=iou, vflag=fd, det = None if det is None else det[k], label=label)
            if fd!='':
                fd.close()
            accs.append(acc)
            anas.append(ana)
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, anas, names

def parseSequences(seqmap):
    assert os.path.isfile(seqmap), 'Seqmap %s not found.'%seqmap
    fd = open(seqmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row=='' or row=='name' or row[0]=='#': continue
        res.append(row)
    fd.close()
    return res

def generateSkippedGT(gtfile, skip, fmt, block = 1):
    tf = NamedTemporaryFile(delete=False, mode='w')
    with open(gtfile) as fd:
        lines = fd.readlines()
        for line in lines:
            arr = line.strip().split(',')
            fr = int(arr[0])
            if skip!=0 and fr%(skip+1)!=1:
                continue
            pos = line.find(',')
            new_fr = fr//(skip+1)+(1 if skip>0 else 0)
            if block!=1 and new_fr%block!=1:
                continue
            newline = str(new_fr) + line[pos:]
            tf.write(newline)
    tf.close()
    tempfile = tf.name
    return tempfile


if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    seqs = parseSequences(args.seqmap)
    gtfiles = [os.path.join(args.groundtruths, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(args.tests, '%s.txt'%i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.'%gtfile)
            exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.'%tsfile)
            exit(1)

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    for seq in seqs:
        logging.info('\t%s'%seq)
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')

    if args.skip>0 and 'mot' in args.fmt:
        for i, gtfile in enumerate(gtfiles):
            gtfiles[i] = generateSkippedGT(gtfile, args.skip, fmt=args.fmt)
    
    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f, fmt=args.fmt), os.path.join(args.groundtruths, seqs[i], 'seqinfo.ini')) ) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f, fmt=args.fmt)) for i, f in enumerate(tsfiles)])    
    if args.detections is not None:
        dsfiles = [os.path.join(args.detections, '%s.txt'%i) for i in seqs]
        for dsfile in dsfiles:
            if not os.path.isfile(dsfile):
                logging.error('ds File %s not found.'%dsfile)
                exit(1)
        for i, dsfile in enumerate(dsfiles):
            dsfiles[i] = generateSkippedGT(dsfile, args.skip, fmt=args.fmt, block = args.block)
        ds = OrderedDict([(seqs[i], mm.io.loadtxt(f, fmt=args.fmt)) for i, f in enumerate(dsfiles)])
    else:
        ds = None

    mh = mm.metrics.create()
    st = time.time()
    accs, analysis, names = compare_dataframes(gt, ts, args.log, 1.-args.iou, det = ds, label=args.label)
    logging.info('adding frames: %.3f seconds.'%(time.time()-st))
    
    logging.info('Running metrics')
    
    if args.detections is None:
        if args.k:
            summary = mh.compute_many(accs, anas = analysis, names=names, metrics=mm.metrics.motk_metrics, generate_overall=True)
        else:
            summary = mh.compute_many(accs, anas = analysis, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    else:
        summary = mh.compute_many(accs, anas = analysis, names=names, metrics=mm.metrics.motplus_metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motplus_metric_names))
    logging.info('Completed')
