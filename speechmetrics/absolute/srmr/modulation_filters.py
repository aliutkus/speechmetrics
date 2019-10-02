# -*- coding: utf-8 -*-
# Copyright 2014 João Felipe Santos, jfsantos@emt.inrs.ca
#
# This file is part of the SRMRpy library, and is licensed under the
# MIT license: https://github.com/jfsantos/SRMRpy/blob/master/LICENSE

from __future__ import division
import numpy as np
import scipy.signal as sig

def make_modulation_filter(w0, Q):
    W0 = np.tan(w0/2)
    B0 = W0/Q
    b = np.array([B0, 0, -B0], dtype=np.float)
    a = np.array([(1 + B0 + W0**2), (2*W0**2 - 2), (1 - B0 + W0**2)], dtype=np.float)
    return b, a

def modulation_filterbank(mf, fs, Q):
    return [make_modulation_filter(w0, Q) for w0 in 2*np.pi*mf/fs]

def compute_modulation_cfs(min_cf, max_cf, n):
    spacing_factor = (max_cf/min_cf)**(1.0/(n-1))
    cfs = np.zeros(n)
    cfs[0] = min_cf
    for k in range(1,n):
        cfs[k] = cfs[k-1]*spacing_factor
    return cfs

def modfilt(F, x):
    y = np.zeros((len(F), len(x)), dtype=np.float)
    for k, f in enumerate(F):
        y[k] = sig.lfilter(f[0], f[1], x)
    return y

