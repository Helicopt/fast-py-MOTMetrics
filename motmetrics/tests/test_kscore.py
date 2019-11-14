from pytest import approx
import numpy as np
import pandas as pd
import motmetrics as mm
import pytest
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')

main_metrics = ['mota', 'idf1', 'num_switches', 'idkf']

def test_kscore_1():
    acc = mm.MOTAccumulator()

    for fr in range(10):
        acc.update([1], ['a' if fr!=5 else 'b'], [0.5], frameid=fr)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.8)
    assert metr['idf1'] == approx(0.9)
    assert metr['num_switches'] == approx(2)
    assert metr['idkr'] == approx( ((9./10)**2 + (1./10)**2)**0.5 )
    assert metr['idkp'] == approx( 9./10 * 9/10. + 1./10 * 1/10. )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#1: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_2():
    acc = mm.MOTAccumulator()

    for fr in range(10):
        acc.update([1], ['a' if fr<4 else ('b' if fr<7 else 'c') ], [0.5], frameid=fr)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.8)
    assert metr['idf1'] == approx(0.4)
    assert metr['num_switches'] == approx(2)
    assert metr['idkr'] == approx( ((4./10)**2 + (3./10)**2 + (3./10)**2)**0.5 )
    assert metr['idkp'] == approx( 4./10*4./10 + 3./10*3./10. + 3./10*3./10 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#2: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_3():
    acc = mm.MOTAccumulator()

    for fr in range(10):
        acc.update([1], ['a' if fr<4 else chr(97+fr-3)], [0.5], frameid=fr)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.4)
    assert metr['idf1'] == approx(0.4)
    assert metr['num_switches'] == approx(6)
    assert metr['idkr'] == approx( ((4./10)**2 + 6* (1./10)**2)**0.5 )
    assert metr['idkp'] == approx( 4./10.*4./10 + 1./10*1./10 * 6 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#3: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_4():
    acc = mm.MOTAccumulator()

    for fr in range(10):
        acc.update([1], [chr(97+fr)], [0.5], frameid=fr)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.1)
    assert metr['idf1'] == approx(0.1)
    assert metr['num_switches'] == approx(9)
    assert metr['idkr'] == approx( (10* (1./10.)**2)**0.5 )
    assert metr['idkp'] == approx( 1./10.*1./10 *10. )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#4: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_5():
    acc = mm.MOTAccumulator()

    for fr in range(10):
        acc.update([1], ['a'], [0.5], frameid=fr)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(1)
    assert metr['idf1'] == approx(1)
    assert metr['num_switches'] == approx(0)
    assert metr['idkr'] == approx( (1**2)**0.5 )
    assert metr['idkp'] == approx( 1. )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#5: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_6():
    acc = mm.MOTAccumulator()

    acc.update([1], [], [], frameid=0)
    acc.update([1], [], [], frameid=1)
    acc.update([1], ['a'], [0.5], frameid=2)
    acc.update([1], ['a'], [0.5], frameid=3)
    acc.update([1], ['a'], [0.5], frameid=4)
    acc.update([1], ['a'], [0.5], frameid=5)
    acc.update([1], ['a'], [0.5], frameid=6)
    acc.update([1], ['b'], [0.5], frameid=7)
    acc.update([1], [], [], frameid=8)
    acc.update([1], [], [], frameid=9)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.5)
    assert metr['idf1'] == approx(0.625)
    assert metr['num_switches'] == approx(1)
    assert metr['idkr'] == approx( ((5./10)**2 + (1./10)**2)**0.5 )
    assert metr['idkp'] == approx( 5./10.*5./6 + 1./10.*1./6 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#6: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_7():
    acc = mm.MOTAccumulator()

    acc.update([1], [], [], frameid=0)
    acc.update([1], [], [], frameid=1)
    acc.update([1], [], [], frameid=2)
    acc.update([1], ['a'], [0.5], frameid=3)
    acc.update([1], ['a'], [0.5], frameid=4)
    acc.update([1], ['a'], [0.5], frameid=5)
    acc.update([1], ['b'], [0.5], frameid=6)
    acc.update([1], [], [], frameid=7)
    acc.update([1], [], [], frameid=8)
    acc.update([1], [], [], frameid=9)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)

    assert metr['mota'] == approx(0.3)
    assert metr['idf1'] == approx(0.428571428571428)
    assert metr['num_switches'] == approx(1)
    assert metr['idkr'] == approx( ((3./10)**2 + (1./10)**2)**0.5 )
    assert metr['idkp'] == approx( 3./10*3./4 + 1./10*1./4 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#7: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_8():
    acc = mm.MOTAccumulator()

    acc.update([1,2], ['a','b'], [[0.,np.nan],[np.nan,0.]], frameid=0)
    acc.update([1,2], ['a','b'], [[0.,np.nan],[np.nan,0.]], frameid=1)
    acc.update([1,2], ['a','b'], [[0.,np.nan],[np.nan,0.]], frameid=2)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=3)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=4)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=5)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=6)
    acc.update([2], ['c'], [0.5], frameid=7)
    acc.update([2], ['c'], [0.5], frameid=8)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)
    assert metr['mota'] == approx(0.875)
    assert metr['idf1'] == approx(0.625)
    assert metr['num_switches'] == approx(2)
    assert metr['idkr'] == approx( ((3./7)**2+(4./10)**2)**0.5 * 7/16. + ((3./13)**2 + (6/9.)**2)**0.5 * 9/16. )
    assert metr['idkp'] == approx( (3./7) * 3./16 + (6./9) * 6/16. + ((3./13)**2 + (4./10.)**2)**0.5 * 7./16 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#8: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_9():
    acc = mm.MOTAccumulator()

    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=0)
    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=1)
    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=2)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=3)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=4)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=5)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=6)
    acc.update([2], ['c'], [0.5], frameid=7)
    acc.update([2], ['c'], [0.5], frameid=8)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)
    assert metr['mota'] == approx(0.875)
    assert metr['idf1'] == approx(0.625)
    assert metr['num_switches'] == approx(2)
    assert metr['idkr'] == approx( ((3./7)**2+(4./7)**2)**0.5 * 7/16. + ((3./9)**2 + (6/9.)**2)**0.5 * 9/16. )
    assert metr['idkp'] == approx( (3./7) * 3./16 + (6./9) * 6/16. + (4./7) * 4./16 + (3./9) * 3./16 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#9: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_10():
    acc = mm.MOTAccumulator()

    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=0)
    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=1)
    acc.update([1,2], ['a','d'], [[0.,np.nan],[np.nan,0.]], frameid=2)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=3)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=4)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=5)
    acc.update([1,2], ['b','c'], [[0.,np.nan],[np.nan,0.]], frameid=6)
    acc.update([2], ['c'], [0.5], frameid=7)
    acc.update([], ['c'], [], frameid=8)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)
    assert metr['mota'] == approx(0.8)
    assert metr['idf1'] == approx(0.580645161)
    assert metr['num_switches'] == approx(2)
    assert metr['num_false_positives'] == approx(1)
    assert metr['idkr'] == approx( ((3./7)**2+(4/7.)**2)**0.5 * 7/15. + ((3/8.)**2 + (5/9.)**2)**0.5 * 8/15. )
    assert metr['idkp'] == approx( (3./7) * 3./16 + (5./9) * 6/16. + (4./7) * 4./16 + (3./8) * 3./16 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#10: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_11():
    acc = mm.MOTAccumulator()

    acc.update([1,2], ['a'], [[0.,],[np.nan,]], frameid=0)
    acc.update([1,2], ['a'], [[0.,],[np.nan,]], frameid=1)
    acc.update([1,2], ['a'], [[0.,],[np.nan,]], frameid=2)
    acc.update([1,2], ['a'], [[0.,],[np.nan,]], frameid=3)
    acc.update([1,2], ['a'], [[np.nan],[0.]], frameid=4)
    acc.update([1,2], ['a'], [[np.nan],[0.]], frameid=5)
    acc.update([1,2], ['a'], [[np.nan],[0.]], frameid=6)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)
    assert metr['mota'] == approx(0.5)
    assert metr['idf1'] == approx(0.380952380952)
    assert metr['num_switches'] == approx(0)
    assert metr['idkr'] == approx( ((4./10)**2)**0.5 * 7/14. + ((3/11.)**2)**0.5 * 7/14. )
    assert metr['idkp'] == approx( ((4./10)**2 + (3./11.)**2)**0.5 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#11: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))
    
def test_kscore_12():
    acc = mm.MOTAccumulator()

    acc.update([1], ['a'], [0.], frameid=0)
    acc.update([1], ['a'], [0.], frameid=1)
    acc.update([1], ['a'], [0.], frameid=2)
    acc.update([1], ['a'], [0.], frameid=3)
    acc.update([1], ['a'], [0.], frameid=4)
    acc.update([2], ['a'], [0.], frameid=5)
    acc.update([2], ['a'], [0.], frameid=6)
    acc.update([2], ['a'], [0.], frameid=7)
    acc.update([2], ['a'], [0.], frameid=8)
    
    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'idkf'], return_dataframe=False, return_cached=True)
    assert metr['mota'] == approx(1)
    assert metr['idf1'] == approx(0.55555555555)
    assert metr['num_switches'] == approx(0)
    assert metr['idkr'] == approx( (5./9) * 5./9 + (4./9.)*4./9 )
    assert metr['idkp'] == approx( ((5./9)**2 + (4./9.)**2)**0.5 )
    assert metr['idkf'] == approx( metr['idkr'] * metr['idkp'] * 2 / (metr['idkr'] + metr['idkp']) )
    print('\nCase#12: '+''.join(['%s: %.4f, '%(k, metr[k]) for k in main_metrics]))

def test_ids():
    acc = mm.MOTAccumulator()

    # No data
    acc.update([], [], [], frameid=0)
    # Match
    acc.update([1, 2], ['a', 'b'], [[1, 0], [0, 1]], frameid=1)
    # Switch also Transfer
    acc.update([1, 2], ['a', 'b'], [[0.4, np.nan], [np.nan, 0.4]], frameid=2)
    # Match
    acc.update([1, 2], ['a', 'b'], [[0, 1], [1, 0]], frameid=3)
    # Ascend (switch)
    acc.update([1, 2], ['b', 'c'], [[1, 0], [0.4, 0.7]], frameid=4)
    # Migrate (transfer)
    acc.update([1, 3], ['b', 'c'], [[1, 0], [0.4, 0.7]], frameid=5)
    # No data
    acc.update([], [], [], frameid=6)

    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics=['motp', 'mota', 'num_predictions', 'num_transfer', 'num_ascend', 'num_migrate'], return_dataframe=False, return_cached=True)
    assert metr['num_matches'] == 7
    assert metr['num_false_positives'] == 0
    assert metr['num_misses'] == 0
    assert metr['num_switches'] == 3
    assert metr['num_transfer'] == 3
    assert metr['num_ascend'] == 1
    assert metr['num_migrate'] == 1
    assert metr['num_detections'] == 10
    assert metr['num_objects'] == 10
    assert metr['num_predictions'] == 10
    assert metr['mota'] == approx(1. - (0 + 0 + 3) / 10)
    assert metr['motp'] == approx(1.6 / 10)

def test_correct_average():
    # Tests what is being depicted in figure 3 of 'Evaluating MOT Performance'
    acc = mm.MOTAccumulator(auto_id=True)

    # No track
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])
    acc.update([1, 2, 3, 4], [], [])

    # Track single
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])
    acc.update([4], [4], [0])

    mh = mm.metrics.create()
    metr = mh.compute(acc, metrics='mota', return_dataframe=False)
    assert metr['mota'] == approx(0.2)

def test_motchallenge_files():
    dnames = [
        'TUD-Campus',
        'TUD-Stadtmitte',
    ]
    
    def compute_motchallenge(dname):
        df_gt = mm.io.loadtxt(os.path.join(dname,'gt.txt'))
        df_test = mm.io.loadtxt(os.path.join(dname,'test.txt'))
        return mm.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)

    accs = [compute_motchallenge(os.path.join(DATA_DIR, d)) for d in dnames]

    # For testing
    # [a.events.to_pickle(n) for (a,n) in zip(accs, dnames)]

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, names=dnames, generate_overall=True)

    print()
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names, formatters=mh.formatters))
    # assert ((summary['num_transfer'] - summary['num_migrate']) == (summary['num_switches'] - summary['num_ascend'])).all() # False assertion
    summary = summary[mm.metrics.motchallenge_metrics[:15]]
    expected = pd.DataFrame([
        [0.557659, 0.729730, 0.451253, 0.582173, 0.941441, 8.0, 1, 6, 1, 13, 150, 7, 7, 0.526462, 0.277201],
        [0.644619, 0.819760, 0.531142, 0.608997, 0.939920, 10.0, 5, 4, 1, 45, 452, 7, 6, 0.564014, 0.345904],
        [0.624296, 0.799176, 0.512211, 0.602640, 0.940268, 18.0, 6, 10, 2, 58, 602, 14, 13, 0.555116, 0.330177],
    ])
    np.testing.assert_allclose(summary, expected, atol=1e-3)
