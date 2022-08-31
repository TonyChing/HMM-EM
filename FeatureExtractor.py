'''
Our Feature Extractor
'''
import numpy as np
from GetMusicFeatures import GetMusicFeatures
def FeatureExtractor(signal,samplerate,transpositon_coefficient=1):
    n=transpositon_coefficient
    a=GetMusicFeatures(signal, samplerate, winlength=0.03)
    f=n*a[0]# pitch vectors
    logf=12*np.log2(f/440)+49
    aa = np.zeros(len(a[1]))
    for i in range(1,(len(a[1])-1)):
        aa[i] = (a[1][i-1]+a[1][i]+a[1][i+1])/3 # averaging the pitch value at three adjacent points 
    delta = 0.8 # threshould of correlation value to identify whether there is noise or humming
    index_tone_set = aa>delta # find the frame index of the noise(silent segments)
    logf[~index_tone_set] = np.random.rand(sum(~index_tone_set))-40 
    # set the pitch value of the silent segments(noise) to a very low value and add Gaussian randomness to them
    logf[index_tone_set]=logf[index_tone_set]-np.mean(logf[index_tone_set])
    # change the pitch value of the humming part from absolute value to differential value of its mean

    return logf