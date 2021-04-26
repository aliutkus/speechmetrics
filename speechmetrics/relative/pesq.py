from .. import Metric


class PESQ(Metric):
    def __init__(self, window, hop=None):
        super(PESQ, self).__init__(name='PESQ', window=window, hop=hop)
        self.mono = True
        self.fixed_rate = 16000

    def test_window(self, audios, rate):
        from pesq import pesq
        if len(audios) != 2:
            raise ValueError('PESQ needs a reference and a test signals.')
        return {'pesq': pesq(rate, audios[1], audios[0], 'wb')}


def load(window, hop=None):
    return PESQ(window, hop)
