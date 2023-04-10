#Import necessary libraries (numpy, matplotlib, and scipy's curve_fit).
import numpy as np
import matplotlib.pyplot as plt

# Constants and initial conditions
m = 1.0
k = 1.0
omega = np.sqrt(k/m)
tmax = 10.0
# Time step sizes
biggest_timestep = 1.0
smaller_timestep = 0.000001

x0 = 0
v0 = 1

def calculate_points(dt):
    """
    Calculate position and velocity of harmonic oscillator for a given time step (dt).
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

def calculate_energy_conservation(x, v):
    """
    Calculate energy conservation error for given position (x) and velocity (v) arrays.
    """
    e_0 = 0.5 * k * x0**2 + 0.5 * m * v0**2
    delta_e = 1/len(x) * np.sum(abs((0.5*m*v**2 + 0.5*k*x**2 - e_0)/e_0))
    return delta_e


dt_list = np.logspace(np.log10(smaller_timestep), np.log10(biggest_timestep), 10)
#dt_list = np.linspace(smaller_timestep, biggest_timestep, 20)
delta_e_list = []

# Iterate over time step sizes, calculate positions and velocities, compute energy conservation error, and append to delta_e_list
for dt in dt_list:
    x, v = calculate_points(dt)
    delta_e = calculate_energy_conservation(x, v)
    delta_e_list.append(delta_e)

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

def power_law(x, a, b):
    """
    Power-law function for curve fitting.
    """
    return a * x**b

# Log-transform the data points
log_dt_list = np.log10(dt_list)
log_delta_e_list = np.log10(delta_e_list)

# Perform linear regression on the logged data
coeffs = np.polyfit(log_dt_list, log_delta_e_list, deg=1)
a_log = coeffs[1]
b_log = coeffs[0]

# Convert the coefficients back to the original scale
a = 10 ** a_log
b = b_log

# Create log-log plot
plt.loglog(dt_list, delta_e_list, 'x', label='Data')
plt.loglog(dt_list, power_law(dt_list, a, b), label=f'Fit: $y = {a:.4f}x^{{{b:.4f}}}$')
plt.xlabel('Time step size / s')
plt.ylabel('Delta E / J')
plt.gca().invert_xaxis()
plt.legend()
plt.show()
    


