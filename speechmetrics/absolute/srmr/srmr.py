# -*- coding: utf-8 -*-
# Copyright 2014 João Felipe Santos, jfsantos@emt.inrs.ca
#
# This file is part of the SRMRpy library, and is licensed under the
# MIT license: https://github.com/jfsantos/SRMRpy/blob/master/LICENSE

from __future__ import division
import numpy as np
from scipy.signal import hamming
from .hilbert import hilbert
from .modulation_filters import compute_modulation_cfs, modulation_filterbank,\
                                modfilt
from gammatone.fftweight import fft_gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from srmrpy.segmentaxis import segment_axis

from scipy.io.wavfile import read as readwav


def calc_erbs(low_freq, fs, n_filters):
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1

    erbs = ((centre_freqs(fs, n_filters, low_freq)/ear_q)**order
            + min_bw**order)**(1/order)
    return erbs


def calc_cutoffs(cfs, fs, q):
    # Calculates cutoff frequencies (3 dB) for 2nd order bandpass
    w0 = 2*np.pi*cfs/fs
    B0 = np.tan(w0/2)/q
    L = cfs - (B0 * fs / (2*np.pi))
    R = cfs + (B0 * fs / (2*np.pi))
    return L, R


def normalize_energy(energy, drange=30.0):
    peak_energy = np.max(np.mean(energy, axis=0))
    min_energy = peak_energy*10.0**(-drange/10.0)
    energy[energy < min_energy] = min_energy
    energy[energy > peak_energy] = peak_energy
    return energy


def srmr(x, fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128,
         fast=True, norm=False):
    wLengthS = .256
    wIncS = .064
    # Computing gammatone envelopes
    if fast:
        mfs = 400.0
        gt_env = fft_gtgram(x, fs, 0.010, 0.0025, n_cochlear_filters, low_freq)
    else:
        cfs = centre_freqs(fs, n_cochlear_filters, low_freq)
        fcoefs = make_erb_filters(fs, cfs)
        gt_env = np.abs(hilbert(erb_filterbank(x, fcoefs)))
        mfs = fs

    wLength = int(np.ceil(wLengthS*mfs))
    wInc = int(np.ceil(wIncS*mfs))

    # Computing modulation filterbank with Q = 2 and 8 channels
    mod_filter_cfs = compute_modulation_cfs(min_cf, max_cf, 8)
    MF = modulation_filterbank(mod_filter_cfs, mfs, 2)

    n_frames = int(1 + (gt_env.shape[1] - wLength)//wInc)
    w = hamming(wLength+1)[:-1]  # window is periodic, not symmetric

    energy = np.zeros((n_cochlear_filters, 8, n_frames))
    for i, ac_ch in enumerate(gt_env):
        mod_out = modfilt(MF, ac_ch)
        for j, mod_ch in enumerate(mod_out):
            mod_out_frame = segment_axis(mod_ch, wLength,
                                         overlap=wLength-wInc,
                                         end='pad')
            energy[i, j, :] = np.sum((w*mod_out_frame[:n_frames])**2, axis=1)

    if norm:
        energy = normalize_energy(energy)

    erbs = np.flipud(calc_erbs(low_freq, fs, n_cochlear_filters))

    avg_energy = np.mean(energy, axis=2)
    total_energy = np.sum(avg_energy)

    AC_energy = np.sum(avg_energy, axis=1)
    AC_perc = AC_energy*100/total_energy

    AC_perc_cumsum = np.cumsum(np.flipud(AC_perc))
    K90perc_idx = np.where(AC_perc_cumsum > 90)[0][0]

    BW = erbs[K90perc_idx]

    cutoffs = calc_cutoffs(mod_filter_cfs, fs, 2)[0]

    if (BW > cutoffs[4]) and (BW < cutoffs[5]):
        Kstar = 5
    elif (BW > cutoffs[5]) and (BW < cutoffs[6]):
        Kstar = 6
    elif (BW > cutoffs[6]) and (BW < cutoffs[7]):
        Kstar = 7
    elif (BW > cutoffs[7]):
        Kstar = 8

    return np.sum(avg_energy[:, :4])/np.sum(avg_energy[:, 4:Kstar]), energy


def process_file(f, args):
    fs, s = readwav(f)
    if len(s.shape) > 1:
        s = s[:, 0]
    if np.issubdtype(s.dtype, np.int):
        s = s.astype('float')/np.iinfo(s.dtype).max
    r, energy = srmr(
            s, fs, n_cochlear_filters=args.n_cochlear_filters,
            min_cf=args.min_cf,
            max_cf=args.max_cf,
            fast=args.fast,
            norm=args.norm)
    return f, r


def main():
    import argparse
    import multiprocessing
    import functools

    parser = argparse.ArgumentParser(
        description='Compute the SRMR metric for a given WAV file')
    parser.add_argument(
        '-f', '--fast', dest='fast', action='store_true', default=False,
        help='Use the faster version based on the gammatonegram')
    parser.add_argument(
        '-n', '--norm', dest='norm', action='store_true', default=False,
        help='Use modulation spectrum energy normalization')
    parser.add_argument(
        '--ncochlearfilters', dest='n_cochlear_filters', type=int, default=23,
        help='Number of filters in the acoustic filterbank')
    parser.add_argument(
        '--mincf', dest='min_cf', type=float, default=4.0,
        help='Center frequency of the first modulation filter')
    parser.add_argument(
        '--maxcf', dest='max_cf', type=float, default=128.0,
        help='Center frequency of the last modulation filter')
    parser.add_argument(
        'path', metavar='path', nargs='+',
        help='Path of the file or files to be processed.'
             ' Can also be a folder.')
    args = parser.parse_args()

    if len(args.path) > 1:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        results = dict(p.map(functools.partial(process_file, args=args),
                       args.path))
        for f in args.path:
            print('{}: {}'.format(f, results[f]))
    else:
        f, r = process_file(args.path[0], args)
        print('{}: {}'.format(f, r))


if __name__ == '__main__':
    main()
