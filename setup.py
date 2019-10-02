# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="speechmetrics",
    version="1.0",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'tensorflow-gpu==2.0.0',
        'tqdm',
        'librosa',
        'pystoi',
        'museval',
        'gammatone @ git+https://github.com/detly/gammatone',
        'pypesq @ git+https://github.com/vBaiCai/python-pesq',
        'srmrpy @ git+https://github.com/jfsantos/SRMRpy'
    ],
)
