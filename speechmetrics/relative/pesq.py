from .. import Metric


class PESQ(Metric):
    def __init__(self, window, hop=None):
        super(PESQ, self).__init__(name='PESQ', window=window, hop=hop)
        self.mono = True
        self.fixed_rate = 16000

    def test_window(self, audios, rate):
        from pypesq import pypesq

        if len(audios) != 2:
            raise ValueError('PESQ need§s a reference and a test signals.')

        if rate not in [16000, 8000]:
            raise ValueError("sample rate must be 16000 or 8000 for PESQ ...")

        if rate == 16000:
            return {'pesq': pypesq(fs=rate, ref=audios[1], deg=audios[0], mode='wb')}
        elif rate == 8000:
            return {'pesq': pypesq(fs=rate, ref=audios[1], deg=audios[0], mode='nb')}


def load(window, hop=None):
    return PESQ(window, hop)
