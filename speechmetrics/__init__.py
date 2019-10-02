class Metric:
    def __init__(self, name, window, hop=None, verbose=False):
        # the metric operates on some fixed rate only or only on mono ?
        self.fixed_rate = None
        self.mono = False

        # is the metric absolute or relative ?
        self.absolute = False

        # length and hop of windows
        self.window = window
        if hop is None:
            hop = window
        self.hop = hop
        self.name = name
        self.verbose = verbose

    def test_window(self, audios, rate):
        raise NotImplementedError

    def test(self, *test_files):
        """loading sound files and making sure they all have the same lengths
        (zero-padding to the largest).
        Then, calling the `test_window` function that should be specialised
        depending on the metric."""

        # imports
        import soundfile as sf
        import resampy
        from museval.metrics import Framing
        import numpy as np

        audios = []
        maxlen = 0
        if isinstance(test_files, str):
            test_files = [test_files]
        if self.absolute and len(test_files) > 1:
            if self.verbose:
                print('  [%s] is absolute. Processing first file only'
                      % self.name)
            test_files = [test_files[0],]

        for file in test_files:
            # Loading sound file
            audio, rate = sf.read(file, always_2d=True)
            if self.fixed_rate is not None and rate != self.fixed_rate:
                if self.verbose:
                    print('  [%s] preferred is %dkHz rate. resampling'
                          % (self.name, self.fixed_rate))
                audio = resampy.resample(audio, rate, self.fixed_rate, axis=0)
                rate = self.fixed_rate
            if self.mono and audio.shape[1] > 1:
                if self.verbose:
                    print('  [%s] only supports mono. Will use first channel'
                          % self.name)
                audio = audio[..., 0, None]
            if self.mono:
                audio = audio[..., 0]
            maxlen = max(maxlen, audio.shape[0])
            audios += [audio]

        for index, audio in enumerate(audios):
            if audio.shape[0] != maxlen:
                new = np.zeros((maxlen,) + audio.shape[1:])
                new[:audio.shape[0]] = audio
                audios[index] = new

        if self.window is not None:
            framer = Framing(self.window * rate,
                             self.hop * rate, maxlen)
            nwin = framer.nwin
            result = {}
            for (t, win) in enumerate(framer):
                result_t = self.test_window([audio[win] for audio in audios],
                                            rate)
                for metric in result_t.keys():
                    if metric not in result.keys():
                        result[metric] = np.empty(nwin)
                    result[metric][t] = result_t[metric]
        else:
            result = self.test_window(audios, rate)
        return result


from . import absolute
from . import relative


class MetricsList:
    def __init__(self):
        self.metrics = []

    def __add__(self, metric):
        self.metrics += [metric]
        return self

    def __str__(self):
        return 'Metrics: ' + ' '.join([x.name for x in self.metrics])

    def __call__(self, *files):
        result = {}
        for metric in self.metrics:
            result_metric = metric.test(*files)
            for name in result_metric.keys():
                result[name] = result_metric[name]
        return result


def load(metrics='', window=2, verbose=False):
    """ Load the desired metrics inside a Metrics object that can then
    be called to compute all the desired metrics.

    Parameters:
    ----------
    metrics: str or list of str
        the metrics matching any of these will be automatically loaded. this
        match is relative to the structure of the speechmetrics package.
        For instance:
        * 'absolute' will match all absolute metrics
        * 'absolute.srmr' or 'srmr' will only match SRMR
        * '' will match all

    window: float
        the window length to use for testing the files.

    verbose: boolean
        will display information during computations

    Returns:
    --------

    A MetricsList object, that can be run to get the desired metrics
    """
    import pkgutil
    import importlib

    result = MetricsList()

    found_modules = []
    iterator = pkgutil.walk_packages(__path__, __name__ + '.')

    if isinstance(metrics, str):
        metrics = [metrics]
    for module_info in iterator:
        if any([metric in module_info.name for metric in metrics]):
            module = importlib.import_module(module_info.name)
            if module not in found_modules:
                found_modules += [module],
                if hasattr(module, 'load'):
                    load_function = getattr(module, 'load')
                    new_metric = load_function(window)
                    new_metric.verbose = verbose
                    result += new_metric
                    print('Loaded ', module_info.name)
    return result
