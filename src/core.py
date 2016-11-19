import numpy as np
import scipy
import vamp


""" 
This function receives an audio loaded by libROSA and optional parameters
and returns melodic line calculated from MELODIA.
"""
def calculateMelodicLineMELODIA(audio, sr=44100, params=None):
    # MELODIA
    if params == None:
        params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2,\
              "minpeaksalience": 0.0}
    data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)

    # Hop size and Melody estimated
    hop, melody = data['vector']

    # Reducing a half the size of melody (it has a frequency for each frame) 
    melody = melody[::2] # There are 4 f0s for each frame (discount in internal)
    melody = melody.tolist()
    melody.append(0)
    
    # A clearer option is to get rid of the negative values before plotting
    melody_pos = np.array(melody[:])
    melody_pos[melody_pos<0] = 23000 # Will not appear on spectrogram 
    
    return melody_pos


"""  
This function receives an spectrogram an its melodic line and calculates
its melodic mask.
"""
def generateMelodicMask(D, melody, kind=3, n_harm=20, length=2): 
    # Different masks are calculated depending of kind parameter
    # 1 : Only melodic line f0s
    # 2 : Melodic line and harmonics
    # 3 : Melodic line and harmonics dilated 
    if kind == 1:
        n_harm = 2
        length = 1 
    elif kind == 2:
        length = 1
        
    bins  = int(44100/2048)
    specMelody = np.zeros(D.shape)
    melody = melody[::2] # There are 2 f0s for each frame 

    for i in range(D.shape[1]):
        if melody[i] is not None:
            for k in range(1,n_harm):
                for l in range(-length+1,length+1):
                    position = l+int(k*melody[i]/bins)
                    if position < len(specMelody) and position > 5:
                        specMelody[l+int(k*melody[i]/bins)][i] = 1
            
    return specMelody


def spectralNoveltyFunction(D, spec, gamma=1.0):
    logS = np.log(1+gamma*np.abs(D))

    # Discrete derivative
    logS_1 = logS.copy()
    logS_1[::, 0:-1] = logS[::, 1:]
    x = np.subtract(logS_1, logS)

    # Calculate the novelty function
    snovel = []
    for i in range(x.shape[1]):
        k = x[::,i]
        k = k[k>=0]
        snovel.append(sum(k))

    # Enhanced novelty function
    snovel = snovel - (np.mean(snovel) + np.std(snovel))
    snovel[snovel<=0] = 0
    
    snovel = np.convolve(snovel, np.ones(3, dtype=int), 'same')

    # Normalize the function for every point
    snovel /= np.max(snovel)

    
    # Maximum size of a mask to dilate the spectrum
    maskSize = 50

    # Starting the spectrum
    spectrum = spec.copy()

    # Applying dilation on espectrogram
    for frame in range(int(snovel.shape[0])):
        newSize = int(maskSize * snovel[frame])
        if newSize <= 3:
            newSize = 3
        w, h = 1, newSize
        shape = (w, h)
        mask = np.ones((shape[1],shape[0]))
    
        # Dilate the melodic spectrum
        spectrum[::,frame-w+1:frame+w]\
            = scipy.ndimage.binary_dilation(spec[::,frame-w+1:frame+w],\
                                            structure=mask).astype(int)

    # Cleaning memory
    del snovel, logS, logS_1, x

    return spectrum 

""" This function receives an spectrogram, and 
"""
def hitMissDilateMask(spec, hit=None, miss=None):
    spectrum = spec.copy()

    find = np.array([[0,0,1,1,1]])
        
    replace = np.array([[1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,1,0,0,0,0,0,0,0,0],\
                        [1,1,1,0,0,0,0,0,0,0],\
                        [1,1,1,1,0,0,0,0,0,0],\
                        [1,1,1,1,1,0,0,0,0,0],\
                        [1,1,1,1,1,1,1,1,1,1],\
                        [1,1,1,1,1,0,0,0,0,0],\
                        [1,1,1,0,0,0,0,0,0,0],\
                        [1,1,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0],\
                        [1,0,0,0,0,0,0,0,0,0]]) 

    if hit is not None:
        find = hit

    if miss is not None:
        replace = miss

    spectrum = scipy.ndimage.morphology.\
               binary_hit_or_miss(spec,structure1=find).astype(int)

    spec1 = scipy.ndimage.binary_dilation\
            (spectrum, structure=replace).astype(int)

    spec2 = np.maximum(spec1,spec)

    return spectrum, spec1, spec2
