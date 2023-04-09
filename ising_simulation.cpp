#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <cmath>
// ONLY NON STANDARD LIBRARY. CHECK IF YOU HAVE IT. See https://www.openmp.org/
#include <omp.h>
#include <cstdint>

using namespace std;
using SpinArray = vector<vector<int>>;

// This should probably be a command line argument
const double J = 1.0;

// Find the nearest neighbours of a spin at position (x,y) using the periodic boundary conditions
vector<int> FindNeighbours(const SpinArray &spinArray, int systemSize, int x, int y)
{
    return {
        spinArray[(x + 1) % systemSize][y],
        spinArray[(x - 1 + systemSize) % systemSize][y],
        spinArray[x][(y + 1) % systemSize],
        spinArray[x][(y - 1 + systemSize) % systemSize]};
}

// Calculate the energy of a spin at position (x,y)
double CalculateEnergyForSpin(const SpinArray &spinArray, int systemSize, int externalField, int x, int y)
{
    int neighbourSum = 0;
    for (int neighbour : FindNeighbours(spinArray, systemSize, x, y))
    {
        neighbourSum += neighbour;
    }
    return -J * spinArray[x][y] * neighbourSum - externalField * spinArray[x][y];
}

// Initialize the spin array
SpinArray InitializeSpinArray(int systemSize)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    SpinArray spinArray(systemSize, vector<int>(systemSize));
    for (int i = 0; i < systemSize; ++i)
    {
        for (int j = 0; j < systemSize; ++j)
        {
            spinArray[i][j] = -1; // Cold start
        }
    }
    return spinArray;
}

// Main Monte Carlo step
void MonteCarloStep(SpinArray &spinArray, mt19937 &gen, uniform_int_distribution<> &dis,
                    uniform_real_distribution<> &dis_real, int systemSize, double temperature,
                    int externalField)
{

    int x = dis(gen);
    int y = dis(gen);

    double e_0 = CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);

    // Flip the spin
    spinArray[x][y] *= -1;

    double e_1 = CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);

    double deltaEnergy = e_1 - e_0;

    if (deltaEnergy > 0 && dis_real(gen) > exp(-deltaEnergy / temperature))
    {
        // Reject the flip
        spinArray[x][y] *= -1;
    }
}

// Save the data to a csv file
void SaveToCSV(const string &filename, const vector<double> &temperatures, const vector<double> &values)
{
    ofstream file(filename);
    file << "Temperature,Value" << endl;

    for (size_t i = 0; i < temperatures.size(); ++i)
    {
        file << temperatures[i] << "," << values[i] << endl;
    }
}

// Calculate the average absolute magnetisation per spin
double AbsMagnetisation(const SpinArray &spinArray, int systemSize)
{
    double magnetisation = 0.0;
    for (int x = 0; x < systemSize; ++x)
    {
        for (int y = 0; y < systemSize; ++y)
        {
            magnetisation += spinArray[x][y];
        }
    }
    // return absolute value of magnetisation
    return abs(magnetisation);
}

// Calculate the average absolute magnetisation
double AverageAbsMagnetisation(const vector<double> &magnetisations)
{
    double sum = 0.0;
    for (double value : magnetisations)
    {
        sum += abs(value);
    }
    return sum / magnetisations.size();
}

// Calculate the energy of the entire lattice
double Energy(const SpinArray &spinArray, int systemSize, int externalField)
{
    double energy = 0.0;
    for (int x = 0; x < systemSize; ++x)
    {
        for (int y = 0; y < systemSize; ++y)
        {
            // factor of 0.5 to avoid double counting
            energy += 0.5 * CalculateEnergyForSpin(spinArray, systemSize, externalField, x, y);
        }
    }
    return energy;
}

// Calculate the heat capaticty of the system
double HeatCapacity(vector<double> &energyPerSpinList, double temperature)
{
    double energySum = 0.0;
    double energySquaredSum = 0.0;
    int total = energyPerSpinList.size();

    for (double energy : energyPerSpinList)
    {
        energySum += energy;
        energySquaredSum += energy * energy;
    }

    double averageEnergy = energySum / total;
    double avgeragEnergySquared = energySquaredSum / total;

    return (avgeragEnergySquared - averageEnergy * averageEnergy) / ((temperature * temperature));
}

// Calculate the magnetic susceptibility of the system
double MagneticSusceptibility(vector<double> &magnetisationList, double &averageMagnetisation, double temperature)
{
    double magnetisationSquaredSum = 0.0;

    int total = magnetisationList.size();

    for (double mag : magnetisationList)
    {
        magnetisationSquaredSum += mag * mag;
    }
    double averageMagnetisationSquared = magnetisationSquaredSum / total;

    return abs(averageMagnetisationSquared - (averageMagnetisation * averageMagnetisation)) / (temperature);
}

// Calculate the physical quantities for the sytem
void CalculatePhysicalQuantities(int systemSize, int mcSweeps, int mcAdjustmentSweeps, int externalField, const vector<double> &temperatures, vector<double> &heatCapacitiesPerSpin, vector<double> &magnetisationsPerSpin, vector<double> &susceptibilitiesPerSpin)
{
    int sampleSize = temperatures.size();

    // Initialize the random number generators and distributions
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, systemSize - 1);
    uniform_real_distribution<> dis_real(0.0, 1.0);

// Loop over all temperatures
// Use OpenMP to parallelize the loop
#pragma omp parallel for
    for (int t = 0; t < sampleSize; ++t)
    {

        double temperature = temperatures[t];
        SpinArray spinArray = InitializeSpinArray(systemSize);
        vector<double> magnetizationList(mcSweeps - mcAdjustmentSweeps);
        vector<double> energyList(mcSweeps - mcAdjustmentSweeps);

        // Monte Carlo sweeps
        for (int sweep = 0; sweep < mcSweeps; ++sweep)
        {
            for (int step = 0; step < systemSize; ++step)
            {
                MonteCarloStep(spinArray, gen, dis, dis_real, systemSize, temperature, externalField);
            }
            // Only save the magnetisation and energy after the adjustment sweeps
            if (sweep >= mcAdjustmentSweeps)
            {
                int index = sweep - mcAdjustmentSweeps;
                magnetizationList[index] = AbsMagnetisation(spinArray, systemSize);
                energyList[index] = Energy(spinArray, systemSize, externalField);
            }
        }
        int latticeSites = systemSize * systemSize;
        double averageMagnetisation = AverageAbsMagnetisation(magnetizationList);
        double averageMagnetisationPerSpin = averageMagnetisation / latticeSites;
        double heatCapacityPerSpin = HeatCapacity(energyList, temperature) / latticeSites;
        double susceptibilityPerSpin = MagneticSusceptibility(magnetizationList, averageMagnetisation, temperature) / latticeSites;

// OpenMP critical section to ensure that the data is written to the correct index
#pragma omp critical
        {
            heatCapacitiesPerSpin[t] = heatCapacityPerSpin;
            magnetisationsPerSpin[t] = averageMagnetisationPerSpin;
            susceptibilitiesPerSpin[t] = susceptibilityPerSpin;
            cout << "Temperature: " << temperature << " Heat Capacity: " << heatCapacityPerSpin << " Magnetisation: " << averageMagnetisation << " Susceptibility: " << susceptibilityPerSpin << endl;
        }
    }
}

// External function to allow a call to the simulation from Python
extern "C"
{
    void run_simulation(int32_t systemSize, int32_t mcSweeps, int32_t mcAdjustmentSweeps, int32_t externalField, int32_t tempSampleSize, double *temperatures, double *heatCapacitiesPerSpin, double *magnetisationsPerSpin, double *susceptibilitiesPerSpin)
    {

        vector<double> heatCapacitiesPerSpinVector(tempSampleSize);
        vector<double> magnetisationsPerSpinVector(tempSampleSize);
        vector<double> susceptibilitiesPerSpinVector(tempSampleSize);

        CalculatePhysicalQuantities(systemSize, mcSweeps, mcAdjustmentSweeps, externalField, vector<double>(temperatures, temperatures + tempSampleSize), heatCapacitiesPerSpinVector, magnetisationsPerSpinVector, susceptibilitiesPerSpinVector);

        for (int32_t i = 0; i < tempSampleSize; ++i)
        {
            heatCapacitiesPerSpin[i] = heatCapacitiesPerSpinVector[i];
            magnetisationsPerSpin[i] = magnetisationsPerSpinVector[i];
            susceptibilitiesPerSpin[i] = susceptibilitiesPerSpinVector[i];
        }
    }
}

// P.S the build command is "g++-12 -std=c++14 -shared -O2 -o ising_simulation_arm64.so -fPIC ising_simulation.cpp -fopenmp"
