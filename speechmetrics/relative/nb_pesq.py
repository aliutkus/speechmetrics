from .. import Metric


class NBPESQ(Metric):
    def __init__(self, window, hop=None):
        super(NBPESQ, self).__init__(name='NBPESQ', window=window, hop=hop)
        self.mono = True
        self.fixed_rate = 16000

    def test_window(self, audios, rate):
        from pypesq import pesq
        if len(audios) != 2:
            raise ValueError('NB_PESQ needs a reference and a test signals.')
        return {'nb_pesq': pesq(audios[1], audios[0], rate)}


def load(window, hop=None):
    return NBPESQ(window, hop)
