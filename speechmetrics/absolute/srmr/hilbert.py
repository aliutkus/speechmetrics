# -*- coding: utf-8 -*-
# Copyright 2014 João Felipe Santos, jfsantos@emt.inrs.ca
#
# This file is part of the SRMRpy library, and is licensed under the
# MIT license: https://github.com/jfsantos/SRMRpy/blob/master/LICENSE

import numpy as np
from numpy.fft import fft, ifft

# This is copied straight from scipy.signal. The reason is that scipy.signal's version
# will always use the fft and ifft functions from fftpack. If you have Anaconda with an MKL
# license, you can install the package mklfft, which will plug the faster MKL FFT functions
# into numpy.

def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.
    The transformation is done along the last axis by default.
    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:
    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y
    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_
    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``np.imag(hilbert(x))``, and the
    original signal from ``np.real(hilbert(x))``.
    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           http://en.wikipedia.org/wiki/Analytic_signal
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
        # Make N multiple of 16 to make sure the transform will be fast
        if N % 16:
            N = int(np.ceil(N/16)*16)
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = fft(x, N, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if len(x.shape) > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[ind]
    y = ifft(Xf * h, axis=axis)
    return y[:x.shape[axis]]
