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

def E_to_speed(E_keV: float) -> float:
    """Return beta = v/c from kinetic energy in keV."""
    E_eV = E_keV * 1e3
    gamma = 1 + (E_eV * q_e) / (m_e * c**2)
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta

def visualize_geometry(geometry, width, a, h, r, shift, mode, filename):
    import meep as mp
    import matplotlib.pyplot as plt
    
    # Create a 2D simulation for visualization (set thickness to near-zero)
    sim = mp.Simulation(
        cell_size=mp.Vector3(width * a, width * h, 0),
        geometry=geometry,
        resolution=20
    )
    
    plt.figure(figsize=(12, 8))
    sim.plot2D()
    plt.title(f"PhC Geometry (xy-plane): {mode}, r={r:.3f}a, shift={shift:.3f}h")
    plt.tight_layout()
    
    # Save to file instead of showing (works in terminal environments)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Geometry visualization saved to {filename}")
    plt.close()