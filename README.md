# speechmetrics

This repository is a wrapper around several freely available implementations of objective metrics for estimating the quality of speech signals. It includes both _relative_ and _absolute_ metrics, which means metrics that do or do not need a reference signal, respectively.


If you find speechmetrics useful, you are welcome to cite the original papers for the corresponding metrics, since this is just a wrapper around the implementations that were kindly provided by the original authors.

> Please let me know if you think of some metric with available python implementation that could be included here!

# Installation
As of our recent tests, installation goes smoothly on ubuntu, but there may be some compiler errors for `pypesq` on iOs.


For cpu usage:
```
pip install numpy
pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]
```

For gpu usage (on the MOSNet)

```
pip install numpy
pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[gpu]
```

# Usage

`speechmetrics` has been designed to be easily used in a modular way. All you need to do is to specify the actual metrics you want to use and it will load them.

The process is to:
1. Load the metrics you want with the `load` function from the root of the package, that takes two arguments:
    * metrics: str or list of str
      the available metrics that match this argument will be automatically loaded. This matching is relative to the structure of the speechmetrics package.
      For instance:
        - 'absolute' will match all absolute metrics
        - 'absolute.srmr' or 'srmr' will only match SRMR
        - '' will match all
    * window: float or None
      gives the length in seconds of the windows on which to compute the actual scores. If None, the whole signals will be considered.  
    ```my_metrics = speechmetrics.load('relative', window=5)```

2. Just call the object returned by `load` with your estimated file (and your reference in case of relative metrics.)    
   ```scores = my_metrics(path_to_estimate, path_to_reference)```  
   Numpy arrays are also supported, but the corresponding sampling rate needs to be specified  
   ```scores = my_metrics(estimate_array, reference_array, rate=sampling_rate)```
> __WARNING__: The convention for relative metrics is to provide __estimate first, and reference second__.  
>  This is the opposite as the general convention.  
>     => The advantage is: you can still call absolute metrics with the same code, they will just ignore the reference.  

## Example
```
# the case of absolute metrics
import speechmetrics
window_length = 5 # seconds
metrics = speechmetrics.load('absolute', window_length)
scores = metrics(path_to_audio_file)

# the case of relative metrics
metrics = speechmetrics.load(['bsseval', 'sisdr'], window_length)
scores = metrics(path_to_estimate_file, path_to_reference)

# mixed case, still works
metrics = speechmetrics.load(['bsseval', 'mosnet'], window_length)
scores = metrics(path_to_estimate_file, path_to_reference)

```

# Available metrics

## Absolute metrics (`absolute`)

### MOSNet (`absolute.mosnet` or `mosnet`)
*dimensionless, higher is better. 0=very bad, 5=very good*

As provided by the authors of [MOSNet: Deep Learning based Objective Assessment for Voice Conversion](https://arxiv.org/abs/1904.08352). Original github [here](https://github.com/lochenchou/MOSNet)
> @article{lo2019mosnet,  
  title={MOSNet: Deep Learning based Objective Assessment for Voice Conversion},  
  author={Lo, Chen-Chou and Fu, Szu-Wei and Huang, Wen-Chin and Wang, Xin and Yamagishi, Junichi and Tsao, Yu and Wang, Hsin-Min},  
  journal={arXiv preprint arXiv:1904.08352},  
  year={2019}
}

### SRMR (`absolute.srmr` or `srmr`)
*dimensionless ratio, higher is better. 0=very bad, 1=very good*

As provided by the [SRMR Toolbox](https://github.com/jfsantos/SRMRpy), implemented by [@jfsantos](https://github.com/jfsantos).

* > @article{falk2010non,  
  title={A non-intrusive quality and intelligibility measure of reverberant and dereverberated speech},  
  author={Falk, Tiago H and Zheng, Chenxi and Chan, Wai-Yip},  
  journal={IEEE Transactions on Audio, Speech, and Language Processing},  
  volume={18},  
  number={7},  
  pages={1766--1774},  
  year={2010},  
}

* > @inproceedings{santos2014updated,  
  title={An updated objective intelligibility   estimation metric for normal hearing listeners under noise and reverberation},  
  author={Santos, Joo F and Senoussaoui, Mohammed and Falk, Tiago H},  
  booktitle={Proc. Int. Workshop Acoust. Signal Enhancement},  
  pages={55--59},  
  year={2014}  
}

* > @article{santos2014updating,  
  title={Updating the SRMR-CI metric for improved intelligibility prediction for cochlear implant users},  
  author={Santos, Jo{\~a}o F and Falk, Tiago H},  
  journal={IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)},  
  volume={22},  
  number={12},  
  pages={2197--2206},  
  year={2014},  
}

## Relative metrics (`relative`)

### BSSEval (`relative.bsseval` or `bsseval`)
*expressed in dB, higher is better.*

As presented in [this](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01766791v2/document) paper and freely available in [the official museval page](https://github.com/sigsep/sigsep-mus-eval), corresponds to BSSEval v4. There are 3 submetrics handled here: SDR, SAR, ISR.

> @InProceedings{SiSEC18,  
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",  
  title="The 2018 Signal Separation Evaluation Campaign",  
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",  
  year="2018",  
  pages="293--305"  
}

### PESQ (`relative.pesq` or `pesq`)
*dimensionless, higher is better. 0=very bad, 5=very good*

As implemented [there](https://github.com/vBaiCai/python-pesq) by [@vBaiCai](https://github.com/vBaiCai).

### STOI (`relative.stoi` or `stoi`)
*dimensionless correlation coefficient, higher is better. 0=very bad, 1=very good*

As implemented by [@mpariente](https://github.com/mpariente) [here](https://github.com/mpariente/pystoi)
* > @inproceedings{taal2010short,  
  title={A short-time objective intelligibility measure for time-frequency weighted noisy speech},  
  author={Taal, Cees H and Hendriks, Richard C and Heusdens, Richard and Jensen, Jesper},  
  booktitle={2010 IEEE International Conference on Acoustics, Speech and Signal Processing},  
  pages={4214--4217},  
  year={2010},  
  organization={IEEE}  
}
* > @article{taal2011algorithm,  
  title={An algorithm for intelligibility prediction of time--frequency weighted noisy speech},  
  author={Taal, Cees H and Hendriks, Richard C and Heusdens, Richard and Jensen, Jesper},  
  journal={IEEE Transactions on Audio, Speech, and Language Processing},  
  volume={19},  
  number={7},  
  pages={2125--2136},  
  year={2011},  
  publisher={IEEE}  
}
* > @article{jensen2016algorithm,  
  title={An algorithm for predicting the intelligibility of speech masked by modulated noise maskers},  
  author={Jensen, Jesper and Taal, Cees H},  
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},  
  volume={24},  
  number={11},  
  pages={2009--2022},  
  year={2016},  
  publisher={IEEE}  
}

### SISDR: Scale-invariant SDR (`relative.sisdr` or `sisdr`) 
*expressed in dB, higher is better.*

As described in the following paper and implemented by 
[@Jonathan-LeRoux](https://github.com/Jonathan-LeRoux) [here](https://github.com/sigsep/bsseval/issues/3#issuecomment-494995846)
* > @article{Roux_2019,  
   title={SDR – Half-baked or Well Done?},  
   ISBN={9781479981311},  
   url={http://dx.doi.org/10.1109/ICASSP.2019.8683855},  
   DOI={10.1109/icassp.2019.8683855},  
   journal={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
   publisher={IEEE},  
   author={Roux, Jonathan Le and Wisdom, Scott and Erdogan, Hakan and Hershey, John R.},  
   year={2019},  
   month={May}  
}
