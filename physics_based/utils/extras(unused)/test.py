import numpy as np
import matplotlib.pyplot as plt

# --- Define helper functions ---

def polarizer(angle: float) -> np.ndarray:
    """
    Create a simplified 3x3 Mueller matrix for an ideal linear polarizer
    oriented at the given angle (in radians). For testing purposes this
    simplified form assumes no depolarization and ignores circular polarization.
    """
    # Note: This is a simplified matrix; your actual Mueller matrix may differ.
    M = np.array([
        [1,             np.cos(2*angle),  np.sin(2*angle)],
        [np.cos(2*angle), np.cos(2*angle)**2, np.cos(2*angle)*np.sin(2*angle)],
        [np.sin(2*angle), np.cos(2*angle)*np.sin(2*angle), np.sin(2*angle)**2]
    ])
    return M

def calcLinearStokes(intensities: np.ndarray, polarizer_angles: np.ndarray) -> np.ndarray:
    """
    Calculate the linear Stokes parameters from synthetic intensities and polarizer angles.
    Here, we assume that the polarizer matrix is as given by the simple `polarizer()` function.
    """
    # Build Mueller matrices for each polarizer angle (we only use the first row for intensity)
    muellers = [polarizer(angle) for angle in polarizer_angles]
    
    # For each angle, we take the 0th row (since intensity = first element of M @ stokes)
    # This gives us a system of equations A @ stokes = intensities
    A = np.stack([M[0] for M in muellers], axis=0)  # Shape: (N, 3)
    
    # Compute the pseudoinverse of A and solve for stokes.
    A_pinv = np.linalg.pinv(A)
    
    # Ensure intensities is a column vector (shape: (N,)) or (N,1)
    intensities = np.array(intensities)
    
    stokes_calc = A_pinv @ intensities
    return stokes_calc

def cvtStokesToDoLP(stokes: np.ndarray) -> float:
    """
    Convert a Stokes vector to Degree of Linear Polarization (DoLP).
    For a 3-component Stokes vector [I, Q, U].
    """
    s0, s1, s2 = stokes[0], stokes[1], stokes[2]
    return np.sqrt(s1**2 + s2**2) / s0

def cvtStokesToAoLP(stokes: np.ndarray) -> float:
    """
    Convert a Stokes vector to Angle of Linear Polarization (AoLP) in radians.
    For a 3-component Stokes vector [I, Q, U].
    """
    s1, s2 = stokes[1], stokes[2]
    # The formula returns angles in radians between 0 and pi.
    return np.mod(0.5 * np.arctan2(s2, s1), np.pi)


# --- Test the pipeline using synthetic data ---

def test_polarization_pipeline():
    # Define a known synthetic Stokes vector.
    # For example, let s0 = 1.0 (intensity), s1 and s2 can be set to simulate some polarization.
    stokes_true = np.array([1.0, 0.3, 0.2])
    print("True Stokes vector:", stokes_true)

    # Define a set of polarizer angles (in radians) over which we simulate the measurements.
    # For example, 0, 45, 90, and 135 degrees.
    angles_deg = [0, 45, 90, 135]
    polarizer_angles = np.deg2rad(angles_deg)
    
    # Simulate measured intensities.
    # For each polarizer angle, the intensity is given by the first component of (M @ stokes_true)
    intensities = []
    for angle in polarizer_angles:
        M = polarizer(angle)
        intensity = (M @ stokes_true)[0]
        intensities.append(intensity)
    intensities = np.array(intensities)
    
    print("Synthetic intensities for polarizer angles", angles_deg, ":", intensities)

    # Calculate the Stokes vector from these synthetic intensities.
    stokes_calc = calcLinearStokes(intensities, polarizer_angles)
    print("Calculated Stokes vector:", stokes_calc)

    # Compute DoLP and AoLP.
    dolp = cvtStokesToDoLP(stokes_calc)
    aolp = cvtStokesToAoLP(stokes_calc)
    
    print("Calculated DoLP:", dolp)
    print("Calculated AoLP (radians):", aolp)
    print("Calculated AoLP (degrees):", np.rad2deg(aolp))

    # Plot intensities versus polarizer angle.
    plt.figure(figsize=(6, 4))
    plt.plot(angles_deg, intensities, 'o-', label="Intensity")
    plt.xlabel("Polarizer Angle (deg)")
    plt.ylabel("Intensity")
    plt.title("Simulated Intensity vs. Polarizer Angle")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_polarization_pipeline()
