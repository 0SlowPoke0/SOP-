import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def pauli_x(n, i):
    I_left = sp.identity(2**i, format='csc')
    X = sp.csc_matrix([[0, 1], [1, 0]])
    I_right = sp.identity(2**(n-i-1), format='csc')
    return sp.kron(I_left, sp.kron(X, I_right), format='csc')

def pauli_z(n, i):
    I_left = sp.identity(2**i, format='csc')
    Z = sp.csc_matrix([[1, 0], [0, -1]])
    I_right = sp.identity(2**(n-i-1), format='csc')
    return sp.kron(I_left, sp.kron(Z, I_right), format='csc')

def hamiltonian_ising(n, J=1.0, h=1.0, periodic=False):
    """Construct the Hamiltonian matrix for the quantum Ising model with N spins."""
    H = sp.csc_matrix((2**n, 2**n))

    # Interaction term: -J * sum sigma_x^i * sigma_x^(i+1)
    for i in range(n - 1):
        H -= J * (pauli_x(n, i) @ pauli_x(n, i+1))

    if periodic and n > 2:
        H -= J * (pauli_x(n, 0) @ pauli_x(n, n-1))

    # Transverse field term: -h * sum sigma_z^i
    for i in range(n):
        H -= h * pauli_z(n, i)

    return H

# Testing
n = 6  # Number of spins
J = 1.0  # Interaction strength
h = 1.0  # Transverse field strength

H = hamiltonian_ising(n, J, h, periodic=True)

eigenvalues, eigenvectors = spla.eigsh(H, k=1, which='SA')

print(f"Lowest energy (ground state energy) for N={n}: {eigenvalues[0]}")
print("Ground state wavefunction:")
print(eigenvectors[:, 0])
