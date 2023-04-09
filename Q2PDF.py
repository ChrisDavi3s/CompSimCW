import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Number of samples
n_samples = 10000

# Define delta values
deltas = [1, 3, 5, 10,20]

for delta in deltas:

    # Generate the random scaling factors, phi
    phi = np.random.uniform(1, 1 + delta, n_samples)

    # With 0.5 probability, invert phi
    phi_to_invert = np.random.rand(n_samples) < 0.5
    phi[phi_to_invert] = 1 / phi[phi_to_invert]

    # Original x values, uniform distribution within [0, 1]
    x = np.random.uniform(0, 1, n_samples)

    # Initialize an empty list to store accepted x' values
    x_prime_accepted = []

    for i in range(n_samples):
        x_prime_candidate = x[i] * phi[i]

        # Reject the move if it is outside the interval [0, 1]
        if 0 <= x_prime_candidate <= 1:
            x_prime_accepted.append(x_prime_candidate)

    # Convert the list to a numpy array for further processing
    x_prime_accepted = np.array(x_prime_accepted)

    # Perform kernel density estimation on the accepted x' values
    kde = gaussian_kde(x_prime_accepted)

    # Define the range for the line plot
    x_range = np.linspace(0, 1, 100)

    # Evaluate the estimated PDF at the specified range
    pdf_estimate = kde(x_range)

    # Plot the estimated PDF for each delta value
    plt.plot(x_range, pdf_estimate, label='Delta = {}'.format(delta))
    
    
    #plt.hist(x_prime_accepted, bins=100, alpha=0.5, label='Delta = {}'.format(delta), density=True)

# Add legend, labels and title to the plot
plt.xlabel('x\'')
plt.ylabel('Probability Density (using KDE)')
plt.title('PDE')
plt.legend()
plt.show()
