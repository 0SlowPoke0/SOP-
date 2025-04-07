using ITensors
using Plots
using StatsBase

# ----------------------------
# Parameters
# ----------------------------
N = 10                # Number of spins
J = 1.0               # Coupling constant
h = 1.0               # Transverse field strength
D_cutoff = 16         # Max bond dimension
nsweeps = 5           # Number of DMRG sweeps

# ----------------------------
# Sweeps setup
# ----------------------------
sweeps = Sweeps(nsweeps)
maxdim!(sweeps, D_cutoff)
cutoff!(sweeps, 1e-10)

# ----------------------------
# Build the Ising Hamiltonian
# ----------------------------
sites = siteinds("S=1/2", N)
ampo = AutoMPO()
for j in 1:N-1
    add!(ampo, -J, "Sz", j, "Sz", j+1)
end
for j in 1:N
    add!(ampo, -h, "Sx", j)
end
H = MPO(ampo, sites)

# ----------------------------
# Initial state (random MPS)
# ----------------------------
psi0 = randomMPS(sites, D_cutoff)

# ----------------------------
# Energy observer to track convergence
# ----------------------------
energies = Float64[]
function observer_func(psi, energy; sweep, kwargs...)
    push!(energies, energy)
end

# ----------------------------
# Run DMRG
# ----------------------------
energy, psi = dmrg(H, psi0, sweeps; observer=observer_func)
println("Ground state energy: $energy")

# ----------------------------
# Plot: Energy convergence
# ----------------------------
plot(energies,
     marker=:circle,
     xlabel="Sweep",
     ylabel="Energy",
     title="DMRG Energy Convergence",
     legend=false)
savefig("energy_convergence.png")

# ----------------------------
# Compute entanglement entropy
# ----------------------------
entropies = Float64[]
for b in 1:N-1
    _, S = svd(psi, b)
    s = diag(S)
    probs = s.^2 ./ sum(s.^2)
    entropy = -sum(p -> p * log(p), filter(p -> p > 0, probs))
    push!(entropies, entropy)
end

# ----------------------------
# Plot: Entanglement Entropy
# ----------------------------
plot(entropies,
     marker=:diamond,
     xlabel="Bond",
     ylabel="Entanglement Entropy",
     title="Entanglement Entropy Profile",
     legend=false)
savefig("entanglement_entropy.png")
