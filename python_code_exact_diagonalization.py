import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
import time


def create_ising_hamiltonian(n_sites, J=1.0, h=1.0):
    """
    Create the Hamiltonian for the quantum Ising model:
    H = -J * sum_i σ^z_i σ^z_{i+1} - h * sum_i σ^x_i

    Args:
        n_sites: Number of spins in the chain
        J: Interaction strength
        h: Transverse field strength

    Returns:
        Hamiltonian as a sparse matrix
    """
    # Dimension of the Hilbert space
    dim = 2**n_sites

    # Define Pauli matrices
    sigma_x = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
    sigma_z = sparse.csr_matrix(np.array([[1, 0], [0, -1]]))
    identity = sparse.csr_matrix(np.eye(2))

    # Initialize Hamiltonian
    H = sparse.csr_matrix((dim, dim), dtype=complex)

    # Add interaction terms -J * sum_i σ^z_i σ^z_{i+1}
    for i in range(n_sites):
        # Periodic boundary conditions
        j = (i + 1) % n_sites

        # Construct σ^z_i σ^z_j
        op_list = [identity] * n_sites
        op_list[i] = sigma_z
        op_list[j] = sigma_z

        term = op_list[0]
        for op in op_list[1:]:
            term = sparse.kron(term, op)

        H -= J * term

    # Add transverse field terms -h * sum_i σ^x_i
    for i in range(n_sites):
        op_list = [identity] * n_sites
        op_list[i] = sigma_x

        term = op_list[0]
        for op in op_list[1:]:
            term = sparse.kron(term, op)

        H -= h * term

    return H


def ground_state(H):
    """
    Calculate the ground state and energy of the Hamiltonian

    Args:
        H: Hamiltonian matrix

    Returns:
        E0: Ground state energy
        psi0: Ground state eigenvector
    """
    # Calculate the lowest eigenvalue and eigenvector
    E0, psi0 = eigsh(H, k=1, which="SA")

    return E0[0], psi0.flatten()


def create_product_state(n_sites, config="all_up"):
    """
    Create a product state as the initial state

    Args:
        n_sites: Number of sites
        config: Configuration of spins ('all_up', 'all_down', 'alternating', 'neel')

    Returns:
        Product state vector
    """
    if config == "all_up":
        # All spins up in z-basis |↑↑↑...⟩
        state = np.zeros(2**n_sites)
        state[0] = 1.0
    elif config == "all_down":
        # All spins down in z-basis |↓↓↓...⟩
        state = np.zeros(2**n_sites)
        state[-1] = 1.0
    elif config == "alternating":
        # Alternating up and down |↑↓↑↓...⟩
        index = sum(2 ** (n_sites - 1 - i) for i in range(0, n_sites, 2))
        state = np.zeros(2**n_sites)
        state[index] = 1.0
    elif config == "neel":
        # Néel state |↑↓↑↓...⟩ (same as alternating for this implementation)
        index = sum(2 ** (n_sites - 1 - i) for i in range(0, n_sites, 2))
        state = np.zeros(2**n_sites)
        state[index] = 1.0
    elif config == "x_basis":
        # Product state in x-basis |++++...⟩
        # Each site is in state |+⟩ = (|↑⟩ + |↓⟩)/√2
        state = np.ones(2**n_sites) / np.sqrt(2**n_sites)
    else:
        raise ValueError(f"Unknown configuration: {config}")

    return state


def time_evolution_operator(H, dt):
    """
    Calculate the time evolution operator U(dt) = exp(-i * H * dt)

    Args:
        H: Hamiltonian matrix
        dt: Time step

    Returns:
        Time evolution operator
    """
    # Convert sparse matrix to dense for expm
    H_dense = H.toarray()
    U = expm(-1j * H_dense * dt)
    return U


def evolve_state(psi, U, steps):
    """
    Evolve a state over time using the time evolution operator

    Args:
        psi: Initial state
        U: Time evolution operator
        steps: Number of time steps

    Returns:
        List of states at each time step
    """
    states = [psi]
    current_state = psi

    for _ in range(steps):
        current_state = U @ current_state
        states.append(current_state)

    return states


def calculate_overlap(psi, phi):
    """
    Calculate the overlap |<psi|phi>|^2 between two states

    Args:
        psi: First state
        phi: Second state

    Returns:
        Overlap value
    """
    inner_product = np.vdot(psi, phi)
    return np.abs(inner_product) ** 2


def calculate_entanglement_entropy(psi, n_sites, subsystem_size=None):
    """
    Calculate the entanglement entropy for a bipartition of the system

    Args:
        psi: State vector
        n_sites: Number of sites
        subsystem_size: Size of the subsystem A (default: half the system)

    Returns:
        Entanglement entropy
    """
    if subsystem_size is None:
        subsystem_size = n_sites // 2

    # Reshape the state vector to a matrix
    psi_matrix = psi.reshape(2**subsystem_size, 2 ** (n_sites - subsystem_size))

    # Compute the reduced density matrix using SVD
    u, s, vh = np.linalg.svd(psi_matrix, full_matrices=False)

    # Square the singular values to get the eigenvalues of the reduced density matrix
    eigenvalues = s**2

    # Keep only non-zero eigenvalues to avoid log(0) errors
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    # Calculate the von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy


def main():
    # Parameters
    n_sites = 8  # Number of spins
    J = 1.0  # Interaction strength
    h = 1.0  # Transverse field strength
    dt = 0.05  # Time step
    total_time = 10.0  # Total simulation time
    steps = int(total_time / dt)

    print(f"Simulating quantum Ising model with {n_sites} sites")
    print(f"Interaction strength J = {J}, transverse field h = {h}")

    # Create the Hamiltonian
    print("Creating Hamiltonian...")
    H = create_ising_hamiltonian(n_sites, J, h)

    # Find the ground state (for reference)
    print("Finding ground state...")
    E0, psi0 = ground_state(H)
    print(f"Ground state energy: {E0:.6f}")

    # Create initial state |Ψ(t=0)⟩ as a product state in the x-basis
    # This state is interesting because it's not an eigenstate of the Hamiltonian
    print("Creating initial state...")
    psi_initial = create_product_state(n_sites, config="x_basis")

    # Create the time evolution operator
    print("Creating time evolution operator...")
    U = time_evolution_operator(H, dt)

    # Time evolution
    print("Performing time evolution...")
    psi_states = evolve_state(psi_initial, U, steps)

    # Calculate overlaps and entanglement entropy
    print("Calculating overlaps and entanglement entropy...")
    times = np.arange(0, total_time + dt, dt)

    # Calculate overlap with initial state
    overlaps = [calculate_overlap(psi_initial, psi_t) for psi_t in psi_states]

    # Calculate entanglement entropy for each time step
    entropies = [calculate_entanglement_entropy(psi_t, n_sites) for psi_t in psi_states]

    # Calculate overlap with ground state
    gs_overlaps = [calculate_overlap(psi0, psi_t) for psi_t in psi_states]

    # Plot results
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(times, overlaps)
    plt.xlabel("Time")
    plt.ylabel("Overlap |<Ψ(0)|Ψ(t)>|²")
    plt.title("Overlap between initial state and time-evolved state")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(times, gs_overlaps)
    plt.xlabel("Time")
    plt.ylabel("Overlap |<GS|Ψ(t)>|²")
    plt.title("Overlap between ground state and time-evolved state")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(times, entropies)
    plt.xlabel("Time")
    plt.ylabel("Entanglement Entropy S(t)")
    plt.title("Entanglement Entropy over time")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("quantum_ising_simulation.png")
    plt.show()

    print("Simulation completed!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
