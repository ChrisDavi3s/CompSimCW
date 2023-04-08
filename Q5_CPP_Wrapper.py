import numpy as np
import ctypes
import matplotlib.pyplot as plt

# Load the shared library
# Had to reinstall python to move from rosetta to arm64 to get this to work
lib = ctypes.CDLL('./ising_simulation_arm64.so')

# Define the running parameters
lattice_sizes = [10]

tempSteps = 100
minimumTemperature = 0
maximumTemperature = 5
mcSweeps = 300000
mcAdjustmentSweeps = 200000
etxField = 0

t_critical = 2.269  # Critical temperature for the 2D Ising model

temperatures = np.linspace(minimumTemperature, maximumTemperature, tempSteps)

def run_simulation(systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures):
    tempSampleSize = len(temperatures)
    heatCapacities = np.empty(tempSampleSize, dtype=np.double)
    magnetisations = np.empty(tempSampleSize, dtype=np.double)
    susceptibilities = np.empty(tempSampleSize, dtype=np.double)
    
    # Call the run_simulation function from the shared library
    lib.run_simulation(ctypes.c_int32(systemSize), ctypes.c_int32(mcSweeps), ctypes.c_int32(mcAdjustmentSweeps),
                       ctypes.c_int32(externalField), ctypes.c_int32(tempSampleSize), temperatures.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       heatCapacities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       magnetisations.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       susceptibilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    
    return heatCapacities, magnetisations, susceptibilities

# Call the run_simulation function with different lattice sizes and plot the results

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
fig.suptitle('Physical Quantities for Ising Model (B=0)', fontsize=14)

def plot_data(ax, x, y, ylabel, label, bottom=False):
    #no line between points
    ax.plot(x, y, label=label, linestyle='None', marker='x', markersize=1)
    if bottom:
        ax.set_xlabel('Temperature')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)


for size in lattice_sizes:
    heatCapacities, magnetisations, susceptibilities = run_simulation(size, mcSweeps, mcAdjustmentSweeps, etxField, temperatures)

    print(f'Finished simulation for L={size}')

    # Get the temperatures from the minTemperature, maxTemperature, and tempSampleSize
    #temperatures = np.linspace(1.0, 4.0, tempSteps)

    # Add the data for the current lattice size to the plots
    plot_data(ax1, temperatures, heatCapacities, 'Heat Capacity', f'L={size}')
    plot_data(ax2, temperatures, magnetisations, 'Magnetisation', f'L={size}')
    plot_data(ax3, temperatures, susceptibilities, 'Susceptibility', f'L={size}', bottom = True)

# Add legends to the plots
ax1.legend()
ax2.legend()
ax3.legend()

#name of the file
name = 'ising_model_B' + str(etxField) + '.png'

# Adjust the layout and display the plots
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(name)
plt.show()
