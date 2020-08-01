from .stoi import STOI


class ESTOI(STOI):
    def __init__(self, *args, **kwargs):
        super(ESTOI, self).__init__(*args, **kwargs, estoi=True)


def load(window, hop=None):
    return ESTOI(window, hop)
