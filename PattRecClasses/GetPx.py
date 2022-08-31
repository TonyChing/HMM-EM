from .GaussD import GaussD
import numpy as np

def GetPx(data, out_gauss_set):
    
    N = len(out_gauss_set)
    T = data.shape[1]
    scaled_factor = np.zeros(T)
    pX = np.zeros([N,T])
    for n in range(N):
        pX[n,:] = out_gauss_set[n].prob(data)
        
    for t in range(T):
        scaled_factor[t] = max(pX[:,t])
        pX[:,t] = pX[:,t]/max(pX[:,t])
    
    
    return pX, scaled_factor