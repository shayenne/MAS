# Get command line arguments
import sys

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

# Vamp plugin - to use Melodia plugin
import vamp

# Print colored on terminal
from termcolor import colored

import core

def main():
    plot = False
    hitmiss = False
    k = 3
    
    if "-t" in sys.argv:
        idx = sys.argv.index("-t")
        if len(sys.argv) <= (idx+1):
            print ("Define a type")
            sys.exit(0)
            
        if sys.argv[idx+1] not in ["1","2","3","4A","4B","5A","5B"]:
            print ("The type {} is not valid.".format(sys.argv[idx+1]))
            sys.exit(0)
            
        else:
            arg = sys.argv[idx+1]
            print ("Calculating mask of type", arg)
            if arg < "3":
                k = int(arg)
        hitmiss = True
        
    if "-p" in sys.argv:
        plot = True
    
    # Parameters to calculate the spectrogram
    # Sample rate
    sr = 44100

    # FFT Window size
    win_length = 2048

    # FFT bins
    fb = sr/win_length

    # Frame size
    fs = win_length/sr 

    # Hop size
    hop_length = win_length/4


    # Get the file path from command line
    audio_path = str(sys.argv[1])
    print ('Loading audio file path', audio_path, '...', end="")

    # Load audio signal to y and sample rate to sr
    audio, sr = librosa.load(audio_path, sr=sr)
   
    # Calculate the complex spectrogram 
    D = librosa.stft(audio)
    print (colored('DONE', 'green'))

    # Calculate harmonic and percussive masks
    print ('Separating audio in harmonic and percussive components...', end="")
    mask_H, mask_P = librosa.decompose.hpss(D, mask=True)
    print (colored('DONE', 'green'))
    
    # MELODIA
    print ('Estimating melodic line using Melodia Plugin...', end="")
    melody = core.calculateMelodicLineMELODIA(audio)
    print (colored('DONE', 'green'))


    print ('Building the melodic mask...', end="")
    specMelody = core.generateMelodicMask(D, melody, kind=k)
    print (colored('DONE', 'green'))
    
        
    # Hit/Miss Dilate mask
    if hitmiss:
        specHit, specHitDilated, specMax = core.hitMissDilateMask(specMelody)
        print ('Using dilated hit miss mask...')
        specMelody = specMax
        
    
    # Plot the result melodic mask
    if plot:
        librosa.display.specshow(specMelody, y_axis='log', x_axis='time')
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
    
    # Generate the accompainment mask
    specAccomp = 1-specMelody

    print ('Generating melodic and accompainment audios...', end="")
    # Generating melodic and accompainment audios 
    M = (D * specMelody.astype(float))
    A = (D * specAccomp.astype(float))
    print (colored('DONE', 'green'))

    # Reconstruct the signals
    audio_melody = librosa.core.istft(M)
    audio_accomp = librosa.core.istft(A)

    # Write in .wav files
    librosa.output.write_wav('output_melody.wav', audio_melody, sr, norm=False)
    librosa.output.write_wav('output_accomp.wav', audio_accomp, sr, norm=False)

    print (colored('Successfully completed =)', 'blue'))
    

if __name__ == "__main__":
    print ('Melody Accompainment Separation | Version to Python 3.5')
    
    if len(sys.argv) < 2:
        print ('Usage default: mas.py <audio-filepath>')
        print ()
        print ('For more execution parameters: python mas.py -h')
        sys.exit()

    elif str(sys.argv[1]) in ['-h', '--help']:
        print ('This program receives a polyphonic audio signal\n',\
               'and creates two audio signals, the melodic content\n',\
               'and the accompaniment content of the input audio.')
        print ()
        print ('Usage:')
        print ('\t python mas.py <path-audio-input> [-t type] [-p]')
        print ()
        print ('This version does not change the type yet... :(')
        print ('There is a new version being developed :)')
        """
        print ('You can define the type of separation:')
        print ()
        print (" 1. Melodic mask with only fundamental frequencies ")
        print (" 2. Melodic mask with fundamental frequencies and harmonics ")
        print (" 3. Melodic mask, like above, dilated of 2-size element ")
        print (" 4. Melodic mask, like above, using spectral novelty function") 
        print ("   A. dilated with original spectrum ")
        print ("   B. dilated with percussive spectrum ")
        print (" 5. Melodic mask with hit/miss algorithm ")
        print ("   A. dilated with original spectrum ")
        print ("   B. dilated with percussive spectrum ")
        """
        sys.exit()

    # Start the separation    
    main()
