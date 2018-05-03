# Print 
from __future__ import print_function

# We'll need numpy for some mathematical operations
import numpy as np

# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')


# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
# Functions of time
import time
# Open annotation file
import csv
# Melodia vamp plugin
import vamp
# Scientific python
import scipy
# Get signal processing functions
import scipy.signal as signal
# Make melodic mask
import src.core as mask
# Use sys argument
import sys

# Parameters to calculate the spectrogram
sr = 44100                  # Sample rate
win_length = 2048           # FFT Window size 
fb = sr/win_length          # FFT bins
fs = win_length/sr          # Frame size 
hop_length = win_length/4   # Hop size


if len(sys.argv) < 2:
    print ("Usage: python melodic_separation.py [input_wav]")
    print ("By default, output has _melody.wav in name at same path")
    sys.exit(1)
audio_path = sys.argv[1]

# Load audio file and calculate its STFT
audio, sr = librosa.load(audio_path, sr=sr)
D = librosa.stft(audio, window=signal.cosine)

# parameter values are specified by providing a dicionary to the 
# optional "parameters" parameter:
params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2, "minpeaksalience": 0.0}

data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)
hop, melody = data['vector']
melody = melody.tolist()
melody= melody[::2] # Devolve f0s para cada frame - aqui reduz 2, na funcao reduz 2
melody.append(0)

# A clearer option is to get rid of the negative values before plotting
melody_pos = melody[:].copy()

for i in range(len(melody_pos)):
    if melody_pos[i] <= 0:
        melody_pos[i] = None

# Generate a melodic mask based on Melodia contours 
specMelodia = mask.generateMelodicMask(D, melody_pos, n_harm=50)

# Find start notes and increment spectral information 
spectrum, specFind, specDilated = mask.hitMissDilateMask(specMelodia)

# Save the melodic audio signal dilated with hit and miss
y_m_d = librosa.core.istft(specDilated.astype(float)*D)
librosa.output.write_wav(audio_path[:-4]+"_melody.wav", y_m_d, sr, norm=False)

