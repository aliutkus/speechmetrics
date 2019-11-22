from .. import Metric
import numpy as np
from numpy.linalg import norm


class SISDR(Metric):
    def __init__(self, window, hop=None):
        super(SISDR, self).__init__(name='SISDR', window=window, hop=hop)
        self.mono = True

    def test_window(self, audios, rate):
        # as provided by @Jonathan-LeRoux and slightly adapted for the case of just one reference
        # and one estimate.
        # see original code here: https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846
        eps = np.finfo(audios[0].dtype).eps
        reference = audios[1].reshape(audios[1].size, 1)
        estimate = audios[0].reshape(audios[0].size, 1)
        Rss = np.dot(reference.T, reference)

        # get the scaling factor for clean sources
        a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

        e_true = a * reference
        e_res = estimate - e_true

        Sss = (e_true**2).sum()
        Snn = (e_res**2).sum()

        return {'sisdr': 10 * np.log10((eps+ Sss)/(eps + Snn))}

def load(window, hop=None):
    return SISDR(window, hop)
