from .. import Metric


class STOI(Metric):
    def __init__(self, window, hop=None):
        super(STOI, self).__init__(name='STOI', window=window, hop=hop)
        self.mono = True

    def test_window(self, audios, rate):
        from pystoi.stoi import stoi
        if len(audios) != 2:
            raise ValueError('STOI needs a reference and a test signals.')

        return {'stoi':stoi(audios[0], audios[1], rate, extended=False)}


def load(window, hop=None):
    return STOI(window, hop)
