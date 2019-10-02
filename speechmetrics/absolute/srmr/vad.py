import numpy as np
from srmrpy.segmentaxis import segment_axis

def simple_energy_vad(x, fs, framelen=0.02, theta_main=30, theta_min=-55):
    '''Simple energy voice activity detection algorithm based on energy
    thresholds as described in Tomi Kinnunen and Padmanabhan Rajan, "A
    practical, self-adaptive voice activity detector for speaker verification
    with noisy telephone and microphone data", ICASSP 2013, Vancouver (NOTE:
    this is the benchmark method, not the method proposed by the authors).
    '''
    # Split signal in frames
    framelen = int(framelen * fs)
    frames = segment_axis(x, length=framelen, overlap=0, end='pad')
    frames_zero_mean = frames - frames.mean(axis=0)
    frame_energy = 10*np.log10(1/(framelen-1) * (frames_zero_mean**2).sum(axis=1) + 1e-6)
    max_energy = max(frame_energy)
    speech_presence = (frame_energy > max_energy - theta_main) & (frame_energy > theta_min)
    x_vad = np.zeros_like(x, dtype=bool)
    for idx, frame in enumerate(frames):
        if speech_presence[idx]:
            x_vad[idx*framelen:(idx+1)*framelen] = True
        else:
            x_vad[idx*framelen:(idx+1)*framelen] = False
    return x[x_vad], x_vad

if __name__ == '__main__':
    import sys
    from scipy.io.wavfile import read as readwav
    from matplotlib import pyplot as plt

    fs, s = readwav(sys.argv[1])
    s  = s.astype('float')/np.iinfo(s.dtype).max
    s_vad, speech_presence = simple_energy_vad(s, fs)

    plt.plot(s)
    plt.plot(s_vad - 1, 'g')
    plt.show()

