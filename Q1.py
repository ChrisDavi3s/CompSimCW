import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 1.0
K = 1.0
OMEGA = np.sqrt(K/M)
TMAX = 10.0
BIGGEST_TIMESTEP = 1.0
SMALLER_TIMESTEP = 0.00001
X0 = 0
V0 = 5

def calculate_points(dt, m=M, k=K, x0=X0, v0=V0, tmax=TMAX):
    """
    Calculate position and velocity of harmonic oscillator for a given time step (dt).

    Args:
        dt (float): time step
        m (float): mass (default: M)
        k (float): spring constant (default: K)
        x0 (float): initial position (default: X0)
        v0 (float): initial velocity (default: V0)
        tmax (float): maximum time (default: TMAX)

    Returns:
        x, v (tuple of np.ndarray): position and velocity arrays
    """
    nsteps = int(tmax/dt)
    x = np.zeros(nsteps)
    v = np.zeros(nsteps)    
    x[0] = x0
    v[0] = v0

    for i in range(1, nsteps):
        x[i] = x[i-1] + v[i-1]*dt + 0.5*(-k/m)*x[i-1]*dt**2
        v[i] = v[i-1] + 0.5*(-k/m)*(x[i-1]+x[i])*dt
    return x, v

def calculate_energy_conservation(x, v, m=M, k=K, x0=X0, v0=V0):
    """
    Calculate energy conservation error for given position (x) and velocity (v) arrays.

    Args:
        x (np.ndarray): position array
        v (np.ndarray): velocity array
        m (float): mass (default: M)
        k (float): spring constant (default: K)
        x0 (float): initial position (default: X0)
        v0 (float): initial velocity (default: V0)

    Returns:
        delta_e (float): energy conservation error
    """
    e_0 = 0.5 * k * x0**2 + 0.5 * m * v0**2
    delta_e = 1/len(x) * np.sum(abs((0.5*m*v**2 + 0.5*k*x**2 - e_0)/e_0))
    return delta_e

def main():
    # Generate a list of time step sizes that are logarithmically spaced between SMALLER_TIMESTEP and BIGGEST_TIMESTEP
    dt_list = np.logspace(np.log10(SMALLER_TIMESTEP), np.log10(BIGGEST_TIMESTEP), 10)
    delta_e_list = []

    # Iterate over time step sizes, calculate positions and velocities, compute energy conservation error, 
    # and append to delta_e_list
    for dt in dt_list:
        x, v = calculate_points(dt)
        delta_e = calculate_energy_conservation(x, v)
        delta_e_list.append(delta_e)

    # Convert time step sizes and energy errors to logarithmic scale
    log_dt_list = np.log10(dt_list)
    log_delta_e_list = np.log10(delta_e_list)

    # Perform a linear regression on the log-log data to fit a power-law
    coeffs = np.polyfit(log_dt_list, log_delta_e_list, deg=1)

    # Extract the intercept and slope from the regression result
    a_log = coeffs[1]
    b_log = coeffs[0]

    # Convert intercept to the original scale to get the 'a' in power-law
    a = 10 ** a_log
    # Slope in log-log scale is equivalent to the 'b' in power-law
    b = b_log

    # Plotting style
    plt.style.use({
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.5,
        "xtick.major.size": 6,
        "xtick.major.width": 1.5,
        "ytick.major.size": 6,
        "ytick.major.width": 1.5,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    })

    plt.loglog(dt_list, delta_e_list, 'x', label='Delta E')
    plt.loglog(dt_list, a * dt_list**b, label=f'Fit: $y = {a:.4f}x^{{{b:.4f}}}$')
    plt.xlabel('Time step size / s')
    plt.ylabel('Delta E / J')
    plt.gca().invert_xaxis()
    plt.yscale("log")  # Set the y-axis scale to logarithmic
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
