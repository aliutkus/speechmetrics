# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="speechmetrics",
    version="1.0",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'resampy',
        'pystoi',
        'museval',
        'gammatone @ git+https://github.com/detly/gammatone',
        'pypesq @ git+https://github.com/vBaiCai/python-pesq',
        'srmrpy @ git+https://github.com/jfsantos/SRMRpy',
        'pesq @ git+https://github.com/ludlows/python-pesq',
    ],
    extras_require={
        'cpu': ['tensorflow==2.0.0', 'librosa'],
        'gpu': ['tensorflow-gpu==2.0.0', 'librosa'],
    },
    include_package_data=True
)
