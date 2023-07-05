
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

setup(
    name="speechmetrics",
    version="1.0",
    packages=find_packages(),

    install_requires=[
        'numpy<1.24',
        'scipy',
        'tqdm',
        'resampy',
        'pystoi',
        'museval',
        'tensorflow>=2.0.0',
        'librosa',
        # This is requred, but srmrpy pull it in,
	    # and there is a pip3 conflict if we have the following
	    # line.
        #'gammatone @ git+https://github.com/detly/gammatone',
        'pypesq @ git+https://github.com/vBaiCai/python-pesq',
        'srmrpy @ git+https://github.com/jfsantos/SRMRpy',
        'pesq @ git+https://github.com/ludlows/python-pesq',
    ],
    include_package_data=True
)
