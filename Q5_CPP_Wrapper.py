import numpy as np
import ctypes
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import glob

# Constants
# Define parameters of the Ising model simulation
LATTICE_SIZES = [25]
TEMP_STEPS = 300
MINIMUM_TEMPERATURE = 0.5
MAXIMUM_TEMPERATURE = 5
MC_SWEEPS = 200000
MC_ADJUSTMENT_SWEEPS = 25000
NUMBER_OF_REPEATS = 100 # Number of times to repeat the simulation for each lattice size (1 for no averaging)
EXTERNAL_FIELD = 0

# Flag to run Ising simulation
RUN_ISING_SIMULATIONS = False

#Flags to plot the physical properties for systems
PLOT_PHYSCIAL_PROPERTIES = True
PLOT_PHYSCIAL_PROPERTIES_FOR_LATTICE_SIZES = [25]
PLOT_PHYSICAL_PROPERTIES_SUCCEPTIBILITY_Y_SCALE_MIN = 0.5e-1

#Flags to run plots for critical temperature estimation
PLOT_CRITICAL_TEMPERATURES = False
USE_GPR = True
PLOT_CRITICAL_TEMPERATURES_FOR_LATTICE_SIZES = [5]

#Fit parameters for the regression approach to Tc estimation
FIT_INITIAL_GUESS = [-0.4, 1.2, 3]
FIT_RANGE_LOWER_BOUND = 1.5
FIT_IGNORED_POINTS_AT_UPPER_END = 4

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

    # Runs the simulation multiple times and averages the results.
    def run_simulation_average(self, systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures):
        tempSampleSize = len(temperatures)
        heatCapacities_avg = np.zeros(tempSampleSize, dtype=np.double)
        magnetisations_avg = np.zeros(tempSampleSize, dtype=np.double)
        susceptibilities_avg = np.zeros(tempSampleSize, dtype=np.double)

        for _ in range(NUMBER_OF_REPEATS):
            print(f"Running simulation for L={systemSize} on repeat {_} ")
            heatCapacities, magnetisations, susceptibilities = self.run_simulation(
                systemSize, mcSweeps, mcAdjustmentSweeps, externalField, temperatures)
            heatCapacities_avg += heatCapacities
            magnetisations_avg += magnetisations
            susceptibilities_avg += susceptibilities

        # Divide by the number of repeats to get the average
        heatCapacities_avg /= NUMBER_OF_REPEATS
        magnetisations_avg /= NUMBER_OF_REPEATS
        susceptibilities_avg /= NUMBER_OF_REPEATS

        self.save_to_csv((heatCapacities_avg, magnetisations_avg, susceptibilities_avg), f"simulation_result_L{systemSize}")
        print(f"Finished simulation for L={systemSize}")
        return heatCapacities_avg, magnetisations_avg, susceptibilities_avg
        
    # Runs the Ising simulations for all defined lattice sizes.
    def run_ising_simulations(self):
        results = []
        for size in LATTICE_SIZES:
            heatCapacities_avg, magnetisations_avg, susceptibilities_avg = self.run_simulation_average(
                size, MC_SWEEPS, MC_ADJUSTMENT_SWEEPS, EXTERNAL_FIELD, self.temperatures)
            results.append((heatCapacities_avg, magnetisations_avg, susceptibilities_avg))
        return results

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

# Class to plot the simulation results.
class IsingPropertyPlotter:
    # Class for plotting the physical properties of the Ising model.

    # Initialise the class with the field strength.
    def __init__(self, field_strength=0):
        self.field_strength = field_strength

    # Function for plotting the physical properties of the Ising model.
    def plot_properties(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
        fig.suptitle(
            f"Physical Quantities for Ising Model (B={self.field_strength})", fontsize=14
        )

        data_files = sorted(glob.glob("simulation_result_*.csv"), key=lambda x: int(x.split('L')[1].split('.')[0]))
        for data_file in data_files:
            size = int(data_file.split('L')[1].split('.')[0])
            if size not in PLOT_PHYSCIAL_PROPERTIES_FOR_LATTICE_SIZES:
                continue
            df = pd.read_csv(data_file)
            temperatures = df["Temperature"].values
            heatCapacities = df["Heat_Capacity"].values
            magnetisations = df["Magnetisation"].values
            susceptibilities = df["Susceptibility"].values
            
            self.create_property_plots(
                size, temperatures, heatCapacities, magnetisations, susceptibilities, ax1, ax2, ax3
            )

        # Create legend with custom marker size
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=l.get_color(), markersize=6) for l in lines], labels)
        
        lines, labels = ax2.get_legend_handles_labels()
        ax2.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=l.get_color(), markersize=6) for l in lines], labels)
        
        lines, labels = ax3.get_legend_handles_labels()
        ax3.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=l.get_color(), markersize=6) for l in lines], labels)
        
        ax3.set_yscale('log')  # set y-axis to logarithmic scale for susceptibility plot
        ax3.set_ylim(bottom=PLOT_PHYSICAL_PROPERTIES_SUCCEPTIBILITY_Y_SCALE_MIN)
        
        name = "ising_model_B" + str(self.field_strength) + ".png"

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(name, dpi=300)
        plt.close()

    # Generic plotter method.
    def plot_data(self, ax, x, y, ylabel, label, bottom=False):
        ax.plot(x, y, label=label, linestyle="None", marker=".", markersize=1.5)
        if bottom:
            ax.set_xlabel(r'Temperature ($J / k_B$)')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    # Create the plots for the physical properties.
    def create_property_plots(self, size, temperatures, heatCapacities, magnetisations, susceptibilities, ax1, ax2, ax3):
        self.plot_data(
            ax1, temperatures, heatCapacities, r'Heat Capacity ($C_H$)', f"L={size}"
        )
        self.plot_data(
            ax2, temperatures, magnetisations, "Magnetisation (M)", f"L={size}"
        )
        self.plot_data(
            ax3, temperatures, susceptibilities, r'Susceptibility ($\chi$)', f"L={size}", bottom=True
        )


class IsingCriticalTemperature:
    # Class for estimating the critical temperature of the Ising model.

    # Initialise the class with the field strength.
    def __init__(self, field_strength=0):
        self.field_strength = field_strength

    # Method to fit function for specific heat
    def specific_heat_fit_func(self, T, A, B, c):
        x = 1 - T / c
        return np.where(x > 0, A - B * np.log(x), np.inf)

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
        peak_index = np.argmax(heatCapacities_sorted_no_nan) - FIT_IGNORED_POINTS_AT_UPPER_END
        T_before_peak = temperatures_sorted_no_nan[:peak_index]
        heatCapacities_before_peak = heatCapacities_sorted_no_nan[:peak_index]

        if len(heatCapacities_before_peak) == 0:
            raise ValueError("No data points found before the peak of heat capacities.")
        initial_guess = FIT_INITIAL_GUESS
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
            label=f"Heat Capacity: L={size}",
            markersize=8,
            linestyle="None",
        )
        plt.plot(
            temperatures_fit,
            self.specific_heat_fit_func(temperatures_fit, *popt),
            "r-",
            linewidth=4,
            label=f"Fit: A={A:.3f}, B={B:.3f}, C={c:.3f}",
        )
        plt.xlabel(r'Temperature ($J / k_B$)', fontsize=18)
        plt.ylabel(r'Heat Capacity ($C_H$)', fontsize=18)
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
            if size not in PLOT_CRITICAL_TEMPERATURES_FOR_LATTICE_SIZES:
                continue
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

class IsingCriticalTemperatureGPR:
    # Class for estimating the critical temperature of the Ising model using GPR.

    # Initialise the class with the field strength.
    def __init__(self, field_strength=0):
        self.field_strength = field_strength

    # Estimates the critical temperature by fitting the specific heat data.
    def estimate_critical_temperature(self, temperatures, heatCapacities, size):
        # Fit a Gaussian process to the data
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        
        kernel = C(1.0, (1e-3, 1e4)) * RBF(5, (0.17, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40)
        gp.fit(temperatures.reshape(-1, 1), heatCapacities)

        # Predict over the temperatures range to find maximum
        temp_pred = np.linspace(temperatures.min(), temperatures.max(), 1000).reshape(-1, 1)
        pred, std = gp.predict(temp_pred, return_std=True)

        estimated_critical_temp = temp_pred[np.argmax(pred)][0]

        # Plot the results
        plt.figure(figsize=(10, 8))
        plt.plot(temperatures, heatCapacities, 'b.', label='Original data')
        plt.plot(temp_pred, pred, 'r-', label='GP fit')
        plt.plot([estimated_critical_temp], [np.max(pred)], 'go', label='Estimated Tc')
        plt.fill_between(temp_pred.flatten(), pred-std, pred+std, color='r', alpha=0.2)

        plt.ylabel(r'Heat Capacity ($C_H$)', fontsize=18)
        plt.title(f'Critical Temperature Estimation (L={size})', fontsize=18)
        plt.xlabel(r'Temperature ($J / k_B$)', fontsize=18)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f'critical_temp_L{size}.png', dpi=300)
        plt.close()

        return estimated_critical_temp

    # Estimates the critical temperature for each lattice size.
    def estimate_and_plot_critical_temperatures(self):
        # Find all the CSV files in the current directory
        data_files = glob.glob("simulation_result_*.csv")
        # Iterate through the data files
        for data_file in data_files:
            # Extract size from file name
            size = int(data_file.split('L')[1].split('.')[0])
            if size not in PLOT_CRITICAL_TEMPERATURES_FOR_LATTICE_SIZES:
                continue
            # Load data from the CSV file
            df = pd.read_csv(data_file)
            temperatures = df["Temperature"].values
            heatCapacities = df["Heat_Capacity"].values

            print(f"Estimating Critical Temp for L={size}")
            estimated_critical_temp = self.estimate_critical_temperature(
                temperatures, heatCapacities, size
            )
            print(
                f"Estimated critical temperature for L={size}: {estimated_critical_temp:.3f}"
            )

if __name__ == "__main__":
    # Create an IsingSimulation instance
    sim = IsingSimulation()
    plotter = IsingPropertyPlotter()

    # Run the Ising model simulation
    if RUN_ISING_SIMULATIONS:
        simulation_results = sim.run_ising_simulations()

    # Create plots for the results
    if PLOT_PHYSCIAL_PROPERTIES:
        plotter.plot_properties()

    # Estimate the critical temperature for each lattice size
    if EXTERNAL_FIELD == 0 and PLOT_CRITICAL_TEMPERATURES:
        if USE_GPR:
            critical_temp_estimator = IsingCriticalTemperatureGPR()
            critical_temp_estimator.estimate_and_plot_critical_temperatures()
        else:
            critical_temp_estimator = IsingCriticalTemperature()
            critical_temp_estimator.estimate_critical_temperature_for_each_lattice()