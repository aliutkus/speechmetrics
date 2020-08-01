from .. import Metric


class STOI(Metric):
    def __init__(self, window, hop=None, estoi=False):
        super(STOI, self).__init__(name='STOI', window=window, hop=hop)
        self.mono = True
        self.estoi = estoi

    def test_window(self, audios, rate):
        from pystoi.stoi import stoi
        if len(audios) != 2:
            raise ValueError('STOI needs a reference and a test signals.')

        return {'stoi':stoi(audios[1], audios[0], rate, extended=self.estoi)}


def load(window, hop=None):
    return STOI(window, hop)
