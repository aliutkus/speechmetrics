import os
import h5py
import scipy
import librosa
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from os.path import join
import random
random.seed(1984)

FS = 16000
FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH=256
WIN_LENGTH=512

# dir
DATA_DIR = './data'
AUDIO_DIR = join(DATA_DIR, 'wav')
BIN_DIR = join(DATA_DIR, 'bin')

def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE): 
    # Loading sound file
    y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.

    # Preemphasis
    #y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                     n_fft=fft_size, 
                     hop_length=HOP_LENGTH, 
                     win_length=WIN_LENGTH,
                     window=scipy.signal.hamming,
                     )

    # magnitude spectrogram
    mag = np.abs(linear) #(1+n_fft/2, T)
    
    # shape in (T, 1+n_fft/2)
    return np.transpose(mag.astype(np.float32))  


def read_list(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def read(file_path):
    
    data_file = h5py.File(file_path, 'r')
    mag_sgram = np.array(data_file['mag_sgram'][:])
    
    timestep = mag_sgram.shape[0]
    mag_sgram = np.reshape(mag_sgram,(1, timestep, SGRAM_DIM))
    
    return {
        'mag_sgram': mag_sgram,
    }   

def pad(array, reference_shape):
    
    result = np.zeros(reference_shape)
    result[:array.shape[0],:array.shape[1],:array.shape[2]] = array

    return result

def data_generator(file_list, bin_root, frame=False, batch_size=1):
    index=0
    while True:
            
        filename = [file_list[index+x].split(',')[0].split('.')[0] for x in range(batch_size)]
        
        for i in range(len(filename)):
            all_feat = read(join(bin_root,filename[i]+'.h5'))
            sgram = all_feat['mag_sgram']

            # the very first feat
            if i == 0:
                feat = sgram
                max_timestep = feat.shape[1]
            else:
                if sgram.shape[1] > feat.shape[1]:
                    # extend all feat in feat
                    ref_shape = [feat.shape[0], sgram.shape[1], feat.shape[2]]
                    feat = pad(feat, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                elif sgram.shape[1] < feat.shape[1]:
                    # extend sgram to feat.shape[1]
                    ref_shape = [sgram.shape[0], feat.shape[1], feat.shape[2]]
                    sgram = pad(sgram, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                else:
                    # same timestep, append all
                    feat = np.append(feat, sgram, axis=0)
        
        mos = [float(file_list[x+index].split(',')[1]) for x in range(batch_size)]
        mos=np.asarray(mos).reshape([batch_size])
        frame_mos = np.array([mos[i]*np.ones([feat.shape[1],1]) for i in range(batch_size)])
            
        index += batch_size  
        # ensure next batch won't out of range
        if index+batch_size >= len(file_list):
            index = 0
            random.shuffle(file_list)
        
        if frame:
            yield feat, [mos, frame_mos]
        else:
            yield feat, [mos]
            
            
def extract_to_h5():
    audio_dir = AUDIO_DIR
    output_dir = BIN_DIR
    
    print('audio dir: {}'.format(audio_dir))
    print('output_dir: {}'.format(output_dir))
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    # get filenames
    files = []
    for f in os.listdir(audio_dir):
        if f.endswith('.wav'):
            files.append(f.split('.')[0])
    
    print('start extracting .wav to .h5, {} files found...'.format(len(files)))
            
    for i in tqdm(range(len(files))):
        f = files[i]
        
        # set audio/visual file path
        audio_file = join(audio_dir, f+'.wav')
        
        # spectrogram
        mag = get_spectrograms(audio_file)
        

        with h5py.File(join(output_dir, '{}.h5'.format(f)), 'w') as hf:
            hf.create_dataset('mag_sgram', data=mag)

            
if __name__ == '__main__':

    extract_to_h5()