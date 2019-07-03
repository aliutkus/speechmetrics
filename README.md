# MOSNet
Implementation of  "MOSNet: Deep Learning based Objective Assessment for Voice Conversion"
https://arxiv.org/abs/1904.08352

# Dependency
tensorflow-gpu 2.0 beta1


### Environment set-up



# Usage

1. `cd ./data` and run `bash download.sh` to download the VCC2018 evaluation results and submitted speech. (downsample the submitted speech might take some times)
2. Run `python mos_results_preprocess.py` to prepare the evaluation results 

(Run `python bootsrap_estimation.py` to do the bootstrap experiment for intrinsic MOS calculation)
4. Run `python utils.py` to extract .wav to .h5
5. Run `python train.py --model CNN-BLSTM` to train a CNN-BLSTM version of MOSNet. 

('CNN', 'BLSTM' or 'CNN-BLSTM' are supported in model.py, as described in paper)
6. Run `python test.py` to test on the pre-trained weights with specified model and weight.



# Evaluation Results

The model is trained on the large listening evaluation results released by the Voice Conversion Challenge 2018.

The listening test results can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3257)

The databases and results (submitted speech) can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3061)
