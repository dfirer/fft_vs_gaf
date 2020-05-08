'''
Create datasets for wav sound files with characterization

Functions:
    

CMSC422 Final Project
@author:
    Danielle Firer
    April 21 2020
'''
from scipy.io import wavfile as wav
from sklearn.model_selection import train_test_split
import os
import random
'''
Create a spectrogram for the given audio sample using fourier transform.
Spectrogram is created by computing the fourier transform of a window of
the sample. The window is moved by the stride to get the time change of
the sound. We wrap each window with the Hanning function to avoid artifacts
produced by clipping the edges of a cycle. We assume a sample rate of 16kHz,
so our frequency bins will range from 0 to 8kHz.

@params:
    samples     - 1D np.array of audio amplitudes of length N
    sample_rate - sampling rate of audio in Hz, default 16 kHz
    overlap_p   - percent each window overlaps the previous,
                  used to calculate size of stride default 50%
    window_ms   - size of window in ms, default 20ms
    eps         - minimum value for spectrogram, default 1e-14

@returns:
    bins     - 1D np.array of time bins for spectrogram corresponding 
               to sample time, length 
                   M = ((N / sample_rate * stride * 0.001) - 1)
    freqs    - 1D np.array of frequency bins from 0 to 8kHz, length
                   K = ((sample_rate * window_ms * 0.001) / 2) + 1
    specgram - 2D np.array of spectrogram, where the value is the
               amplitude of the fourier transform at the corresponding
               frequency and time, size
                   (K, M)

    cats:0, dogs:1      female:0, male:1               
'''

class Sound1:
    def __init__(self, sound, rate, type):
        self.sound = sound
        self.rate= rate
        self.type = type


class Sound:
    def __init__(self, sound, rate):
        self.sound = sound
        self.rate= rate

filepath_animal = "./cat_dog/"
filepath_human = "./female_male/"

##cat dog method returns wavinfo_train, wavinfo_test, type_train, type_test
def cat_dog():
    cats=['test/cats/', 'train/cat/']
    dogs=['test/dog/', 'train/dog/']

    all_d_c_1=[]
    all_d_c=[]
    all_d_c_types=[]
    for c in cats:
        files= os.listdir(filepath_animal+c)
        for f in files:
            rate, wav_sample = wav.read(filepath_animal+c+f)
            all_d_c_1.append(Sound1(wav_sample, rate, 0))
            
    for d in dogs:
        files= os.listdir(filepath_animal+d)
        for f in files:
            rate, wav_sample = wav.read(filepath_animal+d+f)
            all_d_c_1.append(Sound1(wav_sample, rate, 1))

    random.shuffle(all_d_c_1)
    for each in all_d_c_1:
        all_d_c.append(Sound(each.sound, each.rate))
        all_d_c_types.append(each.type)

    #print(all_d_c_types)         
    wavinfo_train, wavinfo_test, type_train, type_test = train_test_split(all_d_c, all_d_c_types, test_size=0.25)
    return [wavinfo_train, wavinfo_test, type_train, type_test]

def female_male():
    folds=['female/', 'male/']

    all_d_c_1=[]
    all_d_c=[]
    all_d_c_types=[]
    for c in range(len(folds)):
        files= os.listdir(filepath_human+folds[c])
        for f in files:
            rate, wav_sample = wav.read(filepath_human+folds[c]+f)
            all_d_c_1.append(Sound1(wav_sample, rate, c))
            
    random.shuffle(all_d_c_1)
    for each in all_d_c_1:
        all_d_c.append(Sound(each.sound, each.rate))
        all_d_c_types.append(each.type)

    print(all_d_c_types)
    print(len(all_d_c_types))    
    wavinfo_train, wavinfo_test, type_train, type_test = train_test_split(all_d_c, all_d_c_types, test_size=0.25)
    return [wavinfo_train, wavinfo_test, type_train, type_test]

