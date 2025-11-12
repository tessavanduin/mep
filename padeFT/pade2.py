import numpy as np
from scipy.interpolate import pade

def padeFT(time, signal, omegas):
    signal_cplx = signal.astype(np.complex128) # cast signal to complex
    Coefficients = np.empty((len(omegas), len(time)), dtype='complex128')
    FT = np.empty((len(omegas)), dtype='complex128')
    for i, omega in enumerate(omegas):
        # For each omega, compute the coefficients
        for j, t in enumerate(time):
            Coefficients[i][j] = signal_cplx[j]*np.exp(-1j*omega*t)
        
        # calculate the pade
        P, Q = pade(Coefficients[i], int((len(time) - 1)/2))

        # FT is the pade evaluated at 1
        FT[i] = P(1)/Q(1)
    
    return FT