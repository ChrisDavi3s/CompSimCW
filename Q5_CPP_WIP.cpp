#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std; 
using SpinArray = vector<vector<int>>;
const double J = 1.0;

vector<int> FindNeighbours(const SpinArray& spinArray,int systemSize, int x, int y) {
    return {
        spinArray[(x + 1) % systemSize][y],
        spinArray[(x - 1 + systemSize) % systemSize][y],
        spinArray[x][(y + 1) % systemSize],
        spinArray[x][(y - 1 + systemSize) % systemSize]
    };
}

double CalculateEnergyForSpin(const SpinArray &spinArray, int systemSize, int externalField, int x, int y) {
    int neighbourSum = 0;
    for (int neighbour : FindNeighbours(spinArray, systemSize, x, y)) {
        neighbourSum += neighbour;
    }
    return -J * spinArray[x][y] * neighbourSum - externalField * spinArray[x][y];
}

SpinArray InitializeSpinArray(int systemSize) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    SpinArray spinArray(systemSize, vector<int>(systemSize));
    for (int i = 0; i < systemSize; ++i) {
        for (int j = 0; j < systemSize; ++j) {
            //spinArray[i][j] = dis(gen) * 2 - 1;
            spinArray[i][j] = -1;
        }
    }
    return spinArray;
}

void MonteCarloStep(SpinArray &spinArray, mt19937 &gen, uniform_int_distribution<> &dis,
                    uniform_real_distribution<> &dis_real, int systemSize, double temperature,
                    int externalField) {

    int x = dis(gen);
    int y = dis(gen);

    double e_0 = CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);

    // Flip the spin
    spinArray[x][y] *= -1;

    double e_1 = CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);

    double deltaEnergy = e_1 - e_0;

    if (deltaEnergy > 0 && dis_real(gen) > exp(-deltaEnergy / temperature)) {
        // Reject the flip
        spinArray[x][y] *= -1;
    }
}

void SaveToCSV(const string& filename, const vector<double>& temperatures, const vector<double>& values) {
    ofstream file(filename);
    file << "Temperature,Value" << endl;

    for (size_t i = 0; i < temperatures.size(); ++i) {
        file << temperatures[i] << "," << values[i] << endl;
    }
}

double Magnetisation(const SpinArray& spinArray, int systemSize) {
    double magnetisation = 0.0;
    for (int x = 0; x < systemSize; ++x) {
        for (int y = 0; y < systemSize; ++y) {
            magnetisation += spinArray[x][y];
        }
    }
    // return absolute value of magnetisation
    return abs(magnetisation) / (systemSize * systemSize);
}

double AverageMagnetisation(const vector<double>& magnetisations) {
    double sum = 0.0;
    for (double value : magnetisations) {
        sum += value;
    }
    return sum / magnetisations.size();
}

double Energy(const SpinArray& spinArray, int systemSize, int externalField) {
    double energy = 0.0;
    for (int x = 0; x < systemSize; ++x) {
        for (int y = 0; y < systemSize; ++y) {
            energy += 0.5 * CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);
        }
    }
    return energy / (systemSize * systemSize);
}

double HeatCapacity(vector<double>& energyPerSpinList, double temperature)
{
    double energy_sum = 0.0;
    double energy_squared_sum = 0.0;
    int num_arrays = energyPerSpinList.size();

    for(double energy : energyPerSpinList) {
        energy_sum += energy;
        energy_squared_sum += energy * energy;
    }

    double avg_energy = energy_sum / num_arrays;
    double avg_energy_squared = energy_squared_sum / num_arrays;

    return (avg_energy_squared - avg_energy * avg_energy) / (temperature * temperature);
}

double MagneticSusceptibility(vector<double>& magnetisationPerSpinList, double& averageMagPerSpin, double temperature) {
    double mag_squared_sum = 0.0;
    int num_arrays = magnetisationPerSpinList.size();

    for(double mag : magnetisationPerSpinList) {
        mag_squared_sum += mag * mag;
    }
    double avg_mag_squared = mag_squared_sum / num_arrays;

    return (avg_mag_squared - averageMagPerSpin * averageMagPerSpin) / temperature;
}

void CalculatePhysicalQuantities(int systemSize, int mcSweeps, int mcAdjustmentSweeps, int externalField, const vector<double>& temperatures, vector<double>& heatCapacities, vector<double>& magnetisations, vector<double>& susceptibilities) {
    int sampleSize = temperatures.size();
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, systemSize - 1);
    uniform_real_distribution<> dis_real(0.0, 1.0);
    
    #pragma omp parallel for
    for (int t = 0; t < sampleSize; ++t) {
        
        double temperature = temperatures[t];
        SpinArray spinArray = InitializeSpinArray(systemSize);
        vector<double> magnetizationPerSpinList(mcSweeps-mcAdjustmentSweeps);
        vector<double> energyPerSpinList(mcSweeps-mcAdjustmentSweeps);

        //printNumberOfSpins(spinArray, systemSize);
 
        for (int sweep = 0; sweep < mcSweeps; ++sweep) {
            for (int step = 0; step < systemSize; ++step) {
                MonteCarloStep(spinArray, gen, dis, dis_real, systemSize, temperature, externalField);
            }

            if (sweep >= mcAdjustmentSweeps) {
                int index = sweep - mcAdjustmentSweeps;
                magnetizationPerSpinList[index] = Magnetisation(spinArray, systemSize);
                energyPerSpinList[index] = Energy(spinArray, systemSize, externalField);
            }
        }

       // printNumberOfSpins(spinArray, systemSize);

        double averageMagPerSpin = AverageMagnetisation(magnetizationPerSpinList);
        double heatCapacity = HeatCapacity(energyPerSpinList, temperature);
        double susceptibility = MagneticSusceptibility(magnetizationPerSpinList, averageMagPerSpin, temperature);

        #pragma omp critical
        {
            heatCapacities[t] = heatCapacity;
            magnetisations[t] = averageMagPerSpin;
            susceptibilities[t] = susceptibility;
            cout << "Temperature: " << temperature << " Heat Capacity: " << heatCapacity << " Magnetisation: " << averageMagPerSpin << " Susceptibility: " << susceptibility << endl;
        }
    }
}

int main() {

    const int LATTICE_SIZE = 25;
    const int SWEEPS = 250000;
    const int MC_STEPS_PER_SWEEP = LATTICE_SIZE * LATTICE_SIZE;
    const int MC_ADJUSTMENT_SWEEPS = 80000;
    
    const int EXT_FIELD = 0; // Add external field strength as a variable
    const int TEMP_SAMPLE_SIZE = 200;
    const double MINIMUM_TEMPERATURE = 1.0;
    const double MAXIMUM_TEMPERATURE = 4.0;

    vector<double> temperatures(TEMP_SAMPLE_SIZE);

    if (TEMP_SAMPLE_SIZE == 1) {
        temperatures[0] = MINIMUM_TEMPERATURE;
    } else {
        double step = (MAXIMUM_TEMPERATURE - MINIMUM_TEMPERATURE) / static_cast<double>(TEMP_SAMPLE_SIZE - 1);
        for (int i = 0; i < TEMP_SAMPLE_SIZE; ++i) {
            temperatures[i] = MINIMUM_TEMPERATURE + i * step;
        }
    }

    vector<double> heatCapacities(TEMP_SAMPLE_SIZE);
    vector<double> magnetisations(TEMP_SAMPLE_SIZE);
    vector<double> susceptibilities(TEMP_SAMPLE_SIZE);

    CalculatePhysicalQuantities(LATTICE_SIZE, SWEEPS, MC_ADJUSTMENT_SWEEPS, EXT_FIELD, temperatures,
         heatCapacities, magnetisations, susceptibilities);

    string filename = ".csv";
    SaveToCSV("heatCapacities" + filename, temperatures, heatCapacities);
    SaveToCSV("magnetisations" + filename, temperatures, magnetisations);
    SaveToCSV("susceptibilities" + filename, temperatures, susceptibilities);

    return 0;
}