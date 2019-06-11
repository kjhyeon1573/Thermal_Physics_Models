"""
    What is Ising model?
    
        Ising model is made to explain phase transition easily.
        
        Consider 2-D lattice of atoms like the below picture.
        
                         Atom state:
            : : : :
         .. + + - + ..    + = +1
         .. - + + - ..    
         .. - + - + ..    - = -1
            : : : :
            
        In Ising model, hamiltonian of the two neighbor atoms is,
            
            H = - J * [S_(i,j) * S_(i',j')],    S_(i,j) is the state of the atom at the point (i,j)
                                                J is a constant.
                                                    J > 0: Ferro-magnetic
                                                    J < 0: Anti ferro-magnetic
                                                    J = 0: Non interacting model
        
        The hamiltonian of a lattice point is
        
            H = - J * S_(i,j) * [S_(i+1,j) + S_(i-1,j) + S_(i,j+1) + S_(i,j-1)]
    
    
    How to calculate phase transition?
    
        Use Metropolis Algorithm. 
            
            0. Set the parameters(J, T) to proper value
            
            1. Pick a point, check the state and calculate the hamiltonian H.
            
            2. And change the state, + to -, - to +. Calculate the hamiltonian H'.
            
            3. Compare two hamiltonians H, H'
        
            4. If H' < H, state remains the flipped state.
            
            5. If H' > H, state return the unflipped state by probability exp[-(H-H')/kT]
               or remains remains the flipped state. (k is boltzmann constant, T is temperature)
        
            6. Repeat 1-5.
            
"""

## import modules
import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt


## Case of initial state
def up_state(shape):
    
    # All of initial atom states set to +
    n_state = np.zeros(shape)+1
    
    return n_state

def down_state(shape):
    
    # All of initial atom states set to -
    n_state = np.zeros(shape)-1
    
    return n_state

def random_state(shape):
    
    # All of initial atom states set to randomly
    n_state = np.random.randint(2, size=shape)-2
    
    return n_state


## Choose a point on the lattice
def rand2D_indice(state):
    
    # Make indice list
    X_indice = np.transpose(
               np.tile(np.arange(state.shape[0]),
                      (state.shape[1],1)))
    Y_indice = np.tile(np.arange(state.shape[1]),
                      (state.shape[0],1))
    P = np.append([X_indice],
                  [Y_indice],
                  axis=0).T
    P = P.reshape(-1,2)

    # Suffle indice
    s = np.arange(len(P))
    rand.shuffle(s)
    P = P[s]

    P = list(map(tuple, P))
    
    return P

## Calculate hamiltonian for an atom at the point (i,j)
def Hamiltonian(n_state, i, j):
    
    # Define hamiltonian 0
    H = 0
    
    # Check the shape of the state
    state_indice = n_state.shape
    
    # Calculate Hamiltonian between the neighbor row atoms
    if (i-1<0):
        H += -n_state[i,j]*n_state[i+1,j]

    elif (i+1>= state_indice[0]):
        H += -n_state[i,j]*n_state[i-1,j]

    else:
        H += -n_state[i,j]*(n_state[i+1,j]+n_state[i-1,j])
        
    # Calculate Hamiltonian between the neighbor column atoms
    if (j-1<0):
        H += -n_state[i,j]*n_state[i,j+1]

    elif (j+1>= state_indice[1]):
        H += -n_state[i,j]*n_state[i,j-1]

    else:
        H += -n_state[i,j]*(n_state[i,j+1]+n_state[i,j-1])
    
    return H



## Set the initial state, parameters
state0 = down_state([200, 200])
iteration = 0

# At the critical temperature, C = J/(k*T)
C = np.log(1+np.sqrt(2))/2

# If you want to input the parameters manually, use below variables.
#J = 
#k = 1.3806488e-23
#T = 
#C = J/(k*T)

# Plot initial state.
fig, ax = plt.subplots(1, 1)
ax.set_axis_off()
plt.set_cmap('gray')
ax.imshow(state0)

# Save figure
plt.savefig('./ising/ising_initial.png')


## Metropolis algorithm for Ising model.
# Times to save fig.
for t in range(100):

    # Iteration in fig.
    for dt in range(10):

        # Randomly shuffled coordinates list
        coord_list= rand2D_indice(state0)
        
        len_C = len(coord_list)
        
        # State transition for all coordinates
        for i in range(len_C):
            
            # Call a random indice in list
            (x, y) = coord_list[i]
            
            # Calculate the hamiltonian.
            H = Hamiltonian(state0, x, y)
            dE = -2*H

            # Decide to flip.
            if dE < 0:
                state0[x, y] = -state0[x, y]

            else:
                p = np.exp(-dE*C) > np.random.random()

                if p:
                    state0[x, y] = -state0[x, y]
        
        # Record iteration
        iteration += 1
    
    # Plot state.
    fig, ax = plt.subplots(1, 1)
    ax.set_axis_off()
    plt.set_cmap('gray')
    ax.imshow(state0)
    
    # Save figure
    plt.savefig('./ising/ising_'+ str(iteration) +'.png')
