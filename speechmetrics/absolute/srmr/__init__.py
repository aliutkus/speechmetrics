from ... import Metric


class SRMR(Metric):
    def __init__(self, window, hop=None):
        super(SRMR, self).__init__(name='SRMR', window=window, hop=hop)

        self.fixed_rate = 16000
        self.mono = True
        self.absolute = True

    def test_window(self, audios, rate):
        return {'srmr': srmr(audios[0], self.fixed_rate, n_cochlear_filters=23,
                             low_freq=125, min_cf=4,
                             max_cf=128, fast=True, norm=False)[0]}


def load(window, hop=None):
    from .srmr import srmr
    return SRMR(window, hop)
