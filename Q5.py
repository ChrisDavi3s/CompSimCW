import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#constants
lattice_size = 40
steps = 10000
#external_field = (0, 0)

#initialising the spin array. Create random spin array based o lattice size
def init_spin_array():
    return np.random.choice([-1, 1], size=(lattice_size, lattice_size))

#calcuating the nearest neighbours with periodic boundary conditions
def find_neighbors(spin_array, lattice_size, x, y): #ehh
    return [spin_array[(x + 1) % lattice_size, y],
            spin_array[(x - 1) % lattice_size, y],
            spin_array[x, (y + 1) % lattice_size],
            spin_array[x, (y - 1) % lattice_size]]

def calculate_energy_for_spin(spin_array, external_field ,x,y):
    return -spin_array[x, y] * sum(find_neighbors(spin_array, lattice_size, x, y)) - external_field * spin_array[x, y]

def monte_carlo(spin_array, temperature, external_field):
    spin_array_list = []
    spin_array_list.append(spin_array)
    
    for i in range(steps):
        #choose a random spin
        x = random.randint(0, lattice_size - 1)
        y = random.randint(0, lattice_size - 1)

        e_0 = calculate_energy_for_spin(spin_array,external_field,x,y)
    
        #flip the spin
        trial_spin = spin_array.copy()
        trial_spin[x, y] *= -1

        #calculate the energy of the configuration
        e_1 = calculate_energy_for_spin(trial_spin, external_field,x,y)

        if e_1 < e_0:
            spin_array = trial_spin
        elif random.random() < np.exp(-(e_1 - e_0) / temperature):
            spin_array = trial_spin
        
        spin_array_list.append(spin_array)
    
    return spin_array_list

#run and plot
spin_array = init_spin_array()
spin_array_list = monte_carlo(spin_array, 1, 0)
#plot the spin array as an animation. Add text for the current frame number

fig, ax = plt.subplots()
ims = []
for i in range(len(spin_array_list)):
    im = ax.imshow(spin_array_list[i], animated=True)
    if i == 0:
        ax.imshow(spin_array_list[0])  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True)

plt.show()
