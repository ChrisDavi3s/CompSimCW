import numpy as np
import ctypes
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import scipy.ndimage.filters as filters

# Constants
LATTICE_SIZES = [40]
TEMP_STEPS = 200
MINIMUM_TEMPERATURE = 0
MAXIMUM_TEMPERATURE = 5
MC_SWEEPS = 100000
MC_ADJUSTMENT_SWEEPS = 10000
EXTERNAL_FIELD = 0

# Class to run the C++ simulation
class IsingSimulation:
    def __init__(self):
        self.lib = ctypes.CDLL('./ising_simulation_arm64.so')
        self.t_critical = 2.269  # Critical temperature for the 2D Ising model
        self.temperatures = np.linspace(
            MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE, TEMP_STEPS)

    def run_simulation(self, systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures):
        tempSampleSize = len(temperatures)
        heatCapacities = np.empty(tempSampleSize, dtype=np.double)
        magnetisations = np.empty(tempSampleSize, dtype=np.double)
        susceptibilities = np.empty(tempSampleSize, dtype=np.double)

        self.lib.run_simulation(ctypes.c_int32(systemSize), ctypes.c_int32(mcSweeps), ctypes.c_int32(mcAdjustmentSweeps),
                                 ctypes.c_int32(externalField), ctypes.c_int32(
                                     tempSampleSize), temperatures.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                 heatCapacities.ctypes.data_as(
                                     ctypes.POINTER(ctypes.c_double)),
                                 magnetisations.ctypes.data_as(
                                     ctypes.POINTER(ctypes.c_double)),
                                 susceptibilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

        return heatCapacities, magnetisations, susceptibilities
    
    def plot_data(self, ax, x, y, ylabel, label, bottom=False):
        ax.plot(x, y, label=label, linestyle='None', marker='x', markersize=0.5)
        if bottom:
            ax.set_xlabel('Temperature')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    def run_ising_simulations(self):
        results = []
        for size in LATTICE_SIZES:
            heatCapacities, magnetisations, susceptibilities = self.run_simulation(
                size, MC_SWEEPS, MC_ADJUSTMENT_SWEEPS, EXTERNAL_FIELD, self.temperatures)
            results.append((heatCapacities, magnetisations, susceptibilities))
        return results

    def create_property_plots(self, results):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
        fig.suptitle(f'Physical Quantities for Ising Model (B={EXTERNAL_FIELD})', fontsize=14)

        for size, (heatCapacities, magnetisations, susceptibilities) in zip(LATTICE_SIZES, results):
            print(f'Finished simulation for L={size}')

            self.plot_data(ax1, self.temperatures, heatCapacities, 'Heat Capacity', f'L={size}')
            self.plot_data(ax2, self.temperatures, magnetisations, 'Magnetisation', f'L={size}')
            self.plot_data(ax3, self.temperatures, susceptibilities, 'Susceptibility', f'L={size}', bottom=True)

        ax1.legend()
        ax2.legend()
        ax3.legend()

        name = 'ising_model_B' + str(EXTERNAL_FIELD) + '.png'

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(name)
        plt.close()

    def estimate_critical_temperature(self, magnetisations):
        # Estimate the critical temperature by finding the inflection point of the magnetisation curve
        # Smooth the magnetisation curve to reduce noise
        smoothed_magnetisations = filters.gaussian_filter1d(magnetisations, sigma=2)
        #Find the second derivative of the magnetisation curve
        second_derivative = np.gradient(np.gradient(smoothed_magnetisations, self.temperatures), self.temperatures)
        #Find the index of the inflection point
        inflection_point_index = np.argmax(np.abs(second_derivative))
        estimated_critical_temp = self.temperatures[inflection_point_index]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(self.temperatures, magnetisations, marker='x', markersize=0.5, linestyle='None', label=f'L={LATTICE_SIZES[-1]}')
        ax.plot(estimated_critical_temp, magnetisations[inflection_point_index], 'ro', label=f'Estimated Tc: {estimated_critical_temp:.3f}')

        ax.set_xlabel('Temperature')
        ax.set_ylabel('Magnetisation')
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax.legend()
        ax.set_title(f'Critical Temperature Estimation (L={LATTICE_SIZES[-1]}, B={EXTERNAL_FIELD})')

        filename = f'critical_temperature_estimation_L{LATTICE_SIZES[-1]}_B{EXTERNAL_FIELD}.png'
        plt.savefig(filename)
        plt.close()

        return estimated_critical_temp


    def estimate_critical_temperature_for_largest_lattice(self, results):
        magnetisations_for_largest_size = results[-1][1]
        estimated_critical_temp = self.estimate_critical_temperature(magnetisations_for_largest_size)
        print(f'Estimated critical temperature: {estimated_critical_temp:.3f}')
        return estimated_critical_temp

if __name__ == '__main__':
    sim = IsingSimulation()
    simulation_results = sim.run_ising_simulations()
    sim.create_property_plots(simulation_results)
    sim.estimate_critical_temperature_for_largest_lattice(simulation_results)