import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
m = 1.0
k = 1.0
omega = np.sqrt(k/m)
tmax = 10.0

# Initial conditions
x0 = -5.0
v0 = 0.0

def calculate_points(dt):

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
    e_0 = 0.5 * k * x0**2 + 0.5 * m * v0**2
    delta_e = 1/len(x) * np.sum(abs((0.5*m*v**2 + 0.5*k*x**2 - e_0)/e_0))
    return delta_e

#calculate the points for several time step sizes
biggest_timestep = 1.0
smaller_timestep = 0.00001
dt_list = np.logspace(np.log10(smaller_timestep), np.log10(biggest_timestep), 10)
delta_e_list = []

#calculate the points for several time step sizes defined abve. Step between values by order of magnitude
for dt in dt_list:
    x, v = calculate_points(dt)
    delta_e = calculate_energy_conservation(x, v)
    delta_e_list.append(delta_e)

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
    return a * x**b

params, _ = curve_fit(power_law, dt_list, delta_e_list)

# Create the log-log plot
plt.loglog(dt_list, delta_e_list, 'x', label='Data')
plt.loglog(dt_list, power_law(dt_list, *params), label=f'Fit: $y = {params[0]:.4f}x^{{{params[1]:.4f}}}$')

plt.xlabel('Time step size')
plt.ylabel('Delta E')
plt.gca().invert_xaxis()
plt.legend()

plt.show()
    

