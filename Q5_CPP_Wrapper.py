import numpy as np
import ctypes
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import glob

# Constants
# Define parameters of the Ising model simulation
LATTICE_SIZES = [5]
TEMP_STEPS = 100
MINIMUM_TEMPERATURE = 1.4
MAXIMUM_TEMPERATURE = 2.7
MC_SWEEPS = 27000
MC_ADJUSTMENT_SWEEPS = 20000
NUMBER_OF_REPEATS = 150
EXTERNAL_FIELD = 0

#Fit parameters
INITIAL_GUESS = [-0.4, 1.2, 3]
FIT_RANGE_LOWER_BOUND = 1.5

# Class to run the C++ simulation
class IsingSimulation:
  
   # Initialize IsingSimulation. Loads the C++ simulation library and defines the temperature range.
    def __init__(self):
        # Load C++ simulation library
        self.lib = ctypes.CDLL("./ising_simulation_arm64.so")
        # Define temperature range for simulation
        self.temperatures = np.linspace(
            MINIMUM_TEMPERATURE, MAXIMUM_TEMPERATURE, TEMP_STEPS
        )

    # Runs the C++ simulation for a single lattice size, returning heat capacities, magnetizations, and susceptibilities.
    def run_simulation(
        self, systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures
    ):
        tempSampleSize = len(temperatures)
        heatCapacities = np.empty(tempSampleSize, dtype=np.double)
        magnetisations = np.empty(tempSampleSize, dtype=np.double)
        susceptibilities = np.empty(tempSampleSize, dtype=np.double)

        self.lib.run_simulation(
            ctypes.c_int32(systemSize),
            ctypes.c_int32(mcSweeps),
            ctypes.c_int32(mcAdjustmentSweeps),
            ctypes.c_int32(externalField),
            ctypes.c_int32(tempSampleSize),
            temperatures.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            heatCapacities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            magnetisations.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            susceptibilities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return heatCapacities, magnetisations, susceptibilities
        
    # Plots a given set of data on a given axis.
    def plot_data(self, ax, x, y, ylabel, label, bottom=False):
        ax.plot(x, y, label=label, linestyle="None", marker=".", markersize=1.5)
        if bottom:
            ax.set_xlabel("Temperature")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    # Runs the simulation multiple times and averages the results.
    def run_simulation_average(self, num_repeats, systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures):
        tempSampleSize = len(temperatures)
        heatCapacities_avg = np.zeros(tempSampleSize, dtype=np.double)
        magnetisations_avg = np.zeros(tempSampleSize, dtype=np.double)
        susceptibilities_avg = np.zeros(tempSampleSize, dtype=np.double)

        for _ in range(num_repeats):
            print(f"Running simulation for L={systemSize} on repeat {_} ")
            heatCapacities, magnetisations, susceptibilities = self.run_simulation(
                systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures)
            heatCapacities_avg += heatCapacities
            magnetisations_avg += magnetisations
            susceptibilities_avg += susceptibilities

        # Divide by the number of repeats to get the average
        heatCapacities_avg /= num_repeats
        magnetisations_avg /= num_repeats
        susceptibilities_avg /= num_repeats

        self.save_to_csv((heatCapacities_avg, magnetisations_avg, susceptibilities_avg), f"simulation_result_L{systemSize}")

        return heatCapacities_avg, magnetisations_avg, susceptibilities_avg
        
    # Runs the Ising simulations for all defined lattice sizes.
    def run_ising_simulations(self, num_repeats):
        results = []
        for size in LATTICE_SIZES:
            heatCapacities_avg, magnetisations_avg, susceptibilities_avg = self.run_simulation_average(
                num_repeats, size, MC_SWEEPS, MC_ADJUSTMENT_SWEEPS, EXTERNAL_FIELD, self.temperatures)
            results.append((heatCapacities_avg, magnetisations_avg, susceptibilities_avg))
        return results

    # Creates and saves the plots of the physical properties for each lattice size
    def create_property_plots(self, results):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
        fig.suptitle(
            f"Physical Quantities for Ising Model (B={EXTERNAL_FIELD})", fontsize=14
        )

        for size, (heatCapacities, magnetisations, susceptibilities) in zip(
            LATTICE_SIZES, results
        ):
            print(f"Finished simulation for L={size}")

            self.plot_data(
                ax1, self.temperatures, heatCapacities, "Heat Capacity", f"L={size}"
            )
            self.plot_data(
                ax2, self.temperatures, magnetisations, "Magnetisation", f"L={size}"
            )
            self.plot_data(
                ax3,
                self.temperatures,
                susceptibilities,
                "Susceptibility",
                f"L={size}",
                bottom=True,
            )

        ax1.legend()
        ax2.legend()
        ax3.legend()

        name = "ising_model_B" + str(EXTERNAL_FIELD) + ".png"

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(name, dpi=300)
        plt.close()

    # Method to fit function for specific heat
    def specific_heat_fit_func(self, T, A, B, c):
        x = 1 - T / c
        return np.where(x > 0, A - B * np.log(x), np.inf)
    
    # Saves the simulation results to a CSV file.
    def save_to_csv(self, results, filename):
        heatCapacities, magnetisations, susceptibilities = results
        data = {
            "Temperature": self.temperatures,
            "Heat_Capacity": heatCapacities,
            "Magnetisation": magnetisations,
            "Susceptibility": susceptibilities,
        }
        df = pd.DataFrame(data)
        df.to_csv(f"{filename}.csv", index=False)

    # Estimates the critical temperature by fitting the specific heat data.
    def estimate_critical_temperature(self, temperatures, heatCapacities, size):
        # Estimate the critical temperature by fitting the specific heat data to the function: A - B * ln(1 - T/c)

        # Combine the temperature and heat capacity data
        combined_data = list(zip(temperatures, heatCapacities))

        # Sort the combined data based on temperature
        sorted_data = sorted(combined_data, key=lambda x: x[0])

        # Remove NaN values from the sorted data
        sorted_data_no_nan = [(T, C) for T, C in sorted_data if not np.isnan(C)]

        # Separate the cleaned data back into individual arrays
        temperatures_sorted_no_nan = np.array([item[0] for item in sorted_data_no_nan])
        heatCapacities_sorted_no_nan = np.array(
            [item[1] for item in sorted_data_no_nan]
        )

        # Find the peak in the cleaned heat capacities
        peak_index = np.argmax(heatCapacities_sorted_no_nan) 
        T_before_peak = temperatures_sorted_no_nan[:peak_index]
        heatCapacities_before_peak = heatCapacities_sorted_no_nan[:peak_index]

        if len(heatCapacities_before_peak) == 0:
            raise ValueError("No data points found before the peak of heat capacities.")
        initial_guess = INITIAL_GUESS
        # Set bounds
        lower_bounds = [-np.inf, -np.inf, -20]
        upper_bounds = [np.inf, np.inf, 20]

        # Perform the curve fitting
        T_in_range = np.array(
            [
                T_value
                for T_value, heat_capacity_value in zip(
                    T_before_peak, heatCapacities_before_peak
                )
                if T_value > FIT_RANGE_LOWER_BOUND
            ]
        )
        heatCapacities_in_range = np.array(
            [
                heat_capacity_value
                for T_value, heat_capacity_value in zip(
                    T_before_peak, heatCapacities_before_peak
                )
                if T_value > FIT_RANGE_LOWER_BOUND
            ]
        )

        # Perform curve fitting with the adjusted initial guess and specified bounds
        popt, _ = curve_fit(
            self.specific_heat_fit_func,
            T_in_range,
            heatCapacities_in_range,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )

        A, B, c = popt
        estimated_critical_temp = c

        # Generate a range of temperatures for creating the line of best fit
        temperatures_fit = np.linspace(FIT_RANGE_LOWER_BOUND, np.max(T_in_range), 1000)

        # Plot the results
        plt.figure(figsize=(10, 8))
        plt.plot(
            temperatures,
            heatCapacities,
            color="teal",
            marker=".",
            label="Data",
            markersize=8,
            linestyle="None",
        )
        plt.plot(
            temperatures_fit,
            self.specific_heat_fit_func(temperatures_fit, *popt),
            "r-",
            linewidth=4,
            label=f"Fit: A={A:.3f}, B={B:.3f}, c={c:.3f}",
        )
        plt.xlabel("Temperature / K", fontsize=18)
        plt.ylabel("Heat Capacity", fontsize=18)
        plt.legend(fontsize=12)
        plt.title(f"Critical Temperature Estimation (L={size})", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"critical_temp_L{size}.png", dpi=300)
        plt.close()

        return estimated_critical_temp

    # Estimates the critical temperature for each lattice size.
    def estimate_critical_temperature_for_each_lattice(self):
        # Find all the CSV files in the current directory
        data_files = glob.glob("simulation_result_*.csv")
        # Iterate through the data files
        for data_file in data_files:
            # Extract size from file name
            size = int(data_file.split('L')[1].split('.')[0])

            # Load data from the CSV file
            df = pd.read_csv(data_file)
            temperatures = df["Temperature"].values
            heatCapacities = df["Heat_Capacity"].values

            print(f"Plotted Critical Temp for L={size}")
            estimated_critical_temp = self.estimate_critical_temperature(
                temperatures, heatCapacities, size
            )
            print(
                f"Estimated critical temperature for L={size}: {estimated_critical_temp:.3f}"
            )

if __name__ == "__main__":
    # Create an IsingSimulation instance
    sim = IsingSimulation()

    # Run the Ising model simulation
    simulation_results = sim.run_ising_simulations(NUMBER_OF_REPEATS)

    # Create plots for the results
    sim.create_property_plots(simulation_results)

    # Estimate the critical temperature for each lattice size
    if EXTERNAL_FIELD == 0:
        sim.estimate_critical_temperature_for_each_lattice()