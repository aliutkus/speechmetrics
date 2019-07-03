import librosa
import os
from os.path import join
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile

BASE_DIR = '.'
AUDIO_DIR = join(BASE_DIR, 'submit')
OUTPUT_DIR = join(BASE_DIR, 'wav')

IN_SR = 22050
OUT_SR = 16000
MAX_BIT = 32767  # for 16bit bitrate

def downsample_librosa(src, dst, in_sr, out_sr):
    y_org, sr = librosa.load(src, sr=in_sr)
    y = librosa.resample(y_org, sr, out_sr)
    librosa.output.write_wav(dst, y, out_sr)
    
def downsample_wavfile(src, dst, in_sr, out_sr):
    y_org, sr = librosa.load(src, sr=in_sr)
    y = librosa.resample(y_org, sr, out_sr)
    y = y * MAX_BIT
    wavfile.write(dst, out_sr, y.astype(np.int16))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# get filenames
files = []
for f in os.listdir(AUDIO_DIR):
    if f.endswith('.wav'):
        files.append(f.split('.')[0])
print('start downsampling .wav, {} files found...'.format(len(files)))

for i in tqdm(range(len(files))):
    f = files[i]
    downsample_wavfile(join(AUDIO_DIR, f+'.wav'), join(OUTPUT_DIR, f+'.wav'), IN_SR, OUT_SR)  

