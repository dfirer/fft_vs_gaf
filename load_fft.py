'''
Create spectrograms for wav sound files

Functions:
    spectrogram(samples)
    load_spec(filename)

CMSC422 Final Project
@author:
    Conor Bergman
    March 14 2020
'''

from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import os

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
Code modified from:
https://towardsdatascience.com/understanding-audio-data-fourier-transform-fft-spectrogram-and-speech-recognition-a4072d228520
@author:
    Kartik Chaudry
    January 19 2020
@modifications:
    Conor Bergman
    March 14 2020
'''
def spectrogram(samples, sample_rate = 16000, overlap_p = 0.5, 
                          window_ms = 20.0, eps = 1e-14):
    # Calculate size of window and stride
    stride_ms = window_ms * overlap_p
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    # See: https://www.tek.com/blog/window-functions-spectrum-analyzers
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Take spectrogram to be log of fft (plus some min value)
    specgram = np.log(fft + eps)
    
    # Bins for spectrogram evenly spaced by stride ms
    bins = np.arange(stride_ms * 0.001, samples.shape[0]/sample_rate, stride_ms * 0.001)
    
    return bins, freqs, specgram

'''
Load sound data from a wav file located at *file_path* create spectrogram
and save spectrogram image to ./specs_imgs/ and data to ./specs/ .

@params:
    file_path - string file path of wav file, format
                    ./data/file_name.wav

@returns:
    save image file to
        ./specs_imgs/file_name.png
    save spectrogram 2D np.array to
        ./specs/file_name.out
'''
def load_spec(file_path):
    # Read wav file data
    rate, wav_sample = wav.read(file_path)
    # Create spectrogram using window of 10 ms with 50% overlap
    bins, freqs, spec = spectrogram(wav_sample, rate)
    
    # Clear plots
    plt.cla()
    # Plot spectrogram
    plt.pcolormesh(bins, freqs, spec, cmap='plasma')

    # Write file data
    filename = file_path.replace("./data/", "").replace(".wav","")
    plt.savefig("./specs_imgs/"+filename+".png")
    np.savetxt("./specs/"+filename+".out", spec)


'''
Create spectrogram for all files in ./data/ directory
'''
if __name__ == '__main__':
    filepath = "./data/"

    files = os.listdir(filepath)
    for file in files:
        load_spec(filepath+file)
    
