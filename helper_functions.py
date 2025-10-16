import numpy as np

c = 299792458
h_bar = 1.054571817e-34
q_e =  1.60217646e-19
m_e = 9.1093837139e-31

def E_to_omega(E: float, length_scale: float):
    """Convert energy in eV to frequency in natural units using length scale `length_scale`

    Args:
        E (float): Energy in eV
        length_scale (float): length scale, e.g. 426e-9 m

    Returns:
        float: Frequency omega in N.U.
    """
    return E*q_e*length_scale/(h_bar*2*np.pi*c)

def E_to_speed(E: float):
    """Convert energy in eV to relativistic electron velocity

    Args:
        E (float): Kinetic energy of electron

    Returns:
        float: Velocity in N.U.
    """
    return np.sqrt( 1 - ( 1 / (1 + E*q_e / (m_e*c**2) )**2 ) )