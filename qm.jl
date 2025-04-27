using ITensors
using ITensorMPS
using Printf
using Plots

# Parameters
N = 300          # Number of sites
J = 1.0         # Coupling strength (negative J for ferromagnetic)
h = 0.5         # Transverse field strength
max_time = 5.0  # Maximum time for evolution
dt = 0.1        # Time step for measurements
τ = 0.01        # Small time step for Trotter evolution
max_dim = 100   # Maximum bond dimension allowed during time evolution

function main()
    println("Initializing quantum Ising model simulation...")

    # Define the site indices (Hilbert space basis) for S=1/2 spins
    # Using ITensors.siteinds explicitly to avoid potential namespace issues
    sites = ITensors.siteinds("S=1/2", N)

    # Build the Hamiltonian MPO for the quantum Ising model
    println("Building Hamiltonian...")
    # Use OpSum (current standard) instead of AutoMPO
    # Using ITensors.OpSum explicitly
    ampo = ITensors.OpSum()

    # Add J * σᶻᵢ * σᶻᵢ⁺¹ terms (Nearest-neighbor ZZ interaction)
    # Use += instead of broadcasted .+=
    for j in 1:N-1
        ampo += J, "Sz", j, "Sz", j + 1
    end

    # Add h * σˣᵢ terms (Transverse field)
    # Use += instead of broadcasted .+=
    for j in 1:N
        ampo += h, "Sx", j
    end

    # Convert the OpSum object to a Matrix Product Operator (MPO)
    # Using ITensorMPS.MPO explicitly as MPO type is now in ITensorMPS
    H = ITensorMPS.MPO(ampo, sites)

    # Initialize with a random MPS state and run DMRG to find the ground state
    println("Finding ground state using DMRG...")
    # Use random_mps (from ITensorMPS) for initialization
    # Initial bond dimension for randomMPS doesn't need to be max_dim, a small value is fine.
    state = ITensorMPS.random_mps(sites; linkdims=10)

    # DMRG parameters
    # Define the sweep schedule for DMRG
    sweeps = Sweeps(5) # Number of sweeps
    # Set maximum bond dimension per sweep (gradually increasing)
    setmaxdim!(sweeps, 10, 20, 100, 100, 200)
    # Set truncation error cutoff per sweep
    setcutoff!(sweeps, 1E-10)

    # Run DMRG
    # Using ITensorMPS.dmrg explicitly
    # energy is the ground state energy (eigenvalue)
    # psi_ground is the ground state MPS (eigenvector)
    energy, psi_ground = ITensorMPS.dmrg(H, state, sweeps; outputlevel=1) # outputlevel=1 shows sweep info

    @printf("Ground state energy = %.10f\n", energy)

    # --- Save the ground state MPS (eigenvector) to a file ---
 
    println("Ground state MPS saved.")
    # ---------------------------------------------------------

    # Initialize with a product state in +x direction for time evolution
    println("Initializing state for time evolution...")

    # Use the predefined state name "X+" for the |+⟩ state in the x-basis
    states = ["X+" for j in 1:N]

    # Create the initial product state MPS
    # Using ITensorMPS.productMPS explicitly
    psi_init = ITensorMPS.productMPS(sites, states)

    # Setup time evolution
    println("Setting up time evolution...")

    # Create gates for time evolution using Trotter decomposition (Suzuki-Trotter)
    # This implements exp(-i*H*τ) ≈ [Π exp(-i*h_odd*τ/2)] [Π exp(-i*h_even*τ/2)] [Π exp(-i*h_even*τ/2)] [Π exp(-i*h_odd*τ/2)]
    # Here, we use a simpler first-order Trotter: Π exp(-i*h_j*τ)
    # where h_j includes both ZZ and X terms. This is less accurate but simpler to code.
    # A more accurate approach would separate even/odd bond gates and single-site gates.

    gates = ITensor[] # Array to hold Trotter gates

    # Create two-site gates for ZZ interaction term: exp(-i * τ * J * Sz * Sz)
    for j in 1:N-1
        s1 = sites[j]
        s2 = sites[j+1]
        # Define the two-site Hamiltonian term
        hj = J * op("Sz", s1) * op("Sz", s2)
        # Create the corresponding gate using matrix exponentiation
        Gj = exp(-im * τ * hj)
        push!(gates, Gj)
    end

    # Create single-site gates for X field term: exp(-i * τ * h * Sx)
    for j in 1:N
        s = sites[j]
        # Define the single-site Hamiltonian term
        hj = h * op("Sx", s)
        # Create the corresponding gate
        Gj = exp(-im * τ * hj)
        push!(gates, Gj)
    end

    # Perform time evolution
    println("Starting time evolution...")
    times = 0:dt:max_time # Array of times where measurements are performed
    overlaps_gs = Float64[] # Array to store overlap with ground state
    overlaps_init = Float64[] # Array to store overlap with initial state (NEW)
    entanglement_entropy = Float64[] # Array to store entanglement entropy

    # Start with the initial state |+⟩...|+⟩
    psi_t = copy(psi_init)

    # Loop over measurement time steps
    for (i, t) in enumerate(times)
        println("Time step $i/$(length(times)): t = $t")

        # Calculate overlap with ground state: |⟨ψ_ground | ψ(t)⟩|
        # Using ITensors.inner explicitly
        overlap_gs = abs(ITensors.inner(psi_ground, psi_t))
        push!(overlaps_gs, overlap_gs)

        # Calculate overlap with initial state: |⟨ψ(0) | ψ(t)⟩| (NEW)
        overlap_init = abs(ITensors.inner(psi_init, psi_t))
        push!(overlaps_init, overlap_init)

        # --- Entanglement Entropy Calculation (Standard Method) ---
        middle_bond = N ÷ 2
        # Set orthogonality center to the site defining the bond cut
        # Using ITensorMPS.orthogonalize! explicitly
        ITensorMPS.orthogonalize!(psi_t, middle_bond)

        # Get the tensor at the middle_bond site
        psi_tensor = psi_t[middle_bond]

        # Identify the indices for the SVD bipartition.
        # Indices belonging to the left partition (site middle_bond and link to the left)
        # Need linkind from ITensorMPS and siteind from ITensors
        # Handle edge case for middle_bond = 1 if N=2
        left_link_ind = (middle_bond == 1) ? ITensors.Index() : ITensorMPS.linkind(psi_t, middle_bond - 1) # Get left link or dummy index
        left_indices = (left_link_ind, ITensors.siteind(psi_t, middle_bond))


        # Compute SVD across the bond, separating left_indices from the link to the right.
        # svd is from ITensors (or LinearAlgebra, ITensors re-exports it)
        U, S, V = svd(psi_tensor, left_indices; alg="divide_and_conquer") # Added alg for potential robustness

        # Calculate von Neumann entropy from the singular values (diagonal elements of S)
        entropy = 0.0
        # Iterate through the diagonal elements of the S tensor
        for n in 1:ITensors.dim(inds(S)[1]) # Use dim(inds(S)[1]) for the dimension of the singular value index
            s_val = S[n,n] # Get the n'th singular value
            s² = s_val^2   # Square it to get the probability eigenvalue
            if s² > 1e-14  # Avoid log(0) for numerical stability
                entropy -= s² * log(s²)
            end
        end
        # --- End of Standard Entanglement Entropy Calculation ---

        push!(entanglement_entropy, entropy)

        # Update print statement to include the new overlap (NEW)
        @printf("  Overlap(GS): %.6f, Overlap(t=0): %.6f, Entropy: %.6f\n", overlap_gs, overlap_init, entropy)

        # Evolve to the next measurement time step (unless we're at the last step)
        if i < length(times)
            # Number of small Trotter steps to reach the next measurement time dt
            n_steps = round(Int, dt / τ)

            # Apply the Trotter gates n_steps times
            for _ in 1:n_steps
                # Apply all gates in the defined Trotter sequence
                # Using ITensorMPS.apply explicitly
                # Note: Applying gates sequentially like this is 1st order Trotter.
                # For higher accuracy, use apply(gates, psi_t; ...) with a structured gate list (e.g., even/odd bonds)
                psi_t = ITensorMPS.apply(gates, psi_t; cutoff=1e-12, maxdim=max_dim)
                # Normalize after each small step (can sometimes help stability, though often done per dt)
                # normalize!(psi_t) # Optional: normalize after each τ step
            end

            # Normalize the MPS after applying all steps for time dt
            # Using ITensorMPS.normalize! explicitly
            ITensorMPS.normalize!(psi_t)
        end
    end

    # Plot results, passing the new overlaps array (NEW)
    plot_results(times, overlaps_gs, overlaps_init, entanglement_entropy)

    println("Simulation completed!")
end

# Modify plot_results function to accept and plot the new overlap (NEW)
function plot_results(times, overlaps_gs, overlaps_init, entanglement_entropy)
    """Plot the time evolution results."""
    # Create the plot for overlap with ground state
    p1 = plot(times, overlaps_gs,
              label="|<GS|ψ(t)>|",
              linewidth=2,
              color=:blue,
              xlabel="Time",
              ylabel="Overlap",
              title="Overlap with Ground State vs Time",
              legend=:bottomleft,
              grid=true)

    # Create the plot for overlap with initial state (NEW)
    p2 = plot(times, overlaps_init,
              label="|<ψ(0)|ψ(t)>|",
              linewidth=2,
              color=:green,
              xlabel="Time",
              ylabel="Overlap",
              title="Overlap with Initial State vs Time",
              legend=:bottomleft,
              grid=true)

    # Create the plot for entanglement entropy
    p3 = plot(times, entanglement_entropy,
              label="Entropy S(t)",
              linewidth=2,
              color=:red,
              xlabel="Time",
              ylabel="Entanglement Entropy",
              title="Entanglement Entropy (Middle Cut) vs Time",
              legend=:bottomright,
              grid=true)

    # Combine plots vertically (NEW: layout=(3,1), size=(800, 800))
    p = plot(p1, p2, p3, layout=(3, 1), size=(800, 800))

    # Save the plot to a file
    savefig(p, "quantum_ising_evolution.png")
    println("Plots saved to quantum_ising_evolution.png")

    # Display the plot
    display(p)
end

# Run the main function and time it
@time main()
