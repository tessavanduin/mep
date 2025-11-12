import numpy as np
from scipy.interpolate import pade

def padeFT(time, signal, omegas):
    signal_cplx = signal.astype(np.complex128) # cast signal to complex
    Dt = time[1] - time[0]
    FT = np.empty(len(omegas), dtype='complex128')

    P, Q = pade(signal_cplx, int((len(time) - 1)/2))
    
    for i, omega in enumerate(omegas):
        FT[i] = P(np.exp(-1j*omega*Dt)) / Q(np.exp(-1j*omega*Dt))
    
    return FT