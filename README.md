
---

# **DMRG Implementation**

This repository contains implementations of the Density-Matrix Renormalization Group (DMRG) algorithm, along with supporting code for numerical simulations of quantum many-body systems. The project is designed to provide pedagogical insights into DMRG concepts and includes Python and Julia implementations.

---

## **Contents**
- **DMRG_python.ipynb**: A Jupyter Notebook showcasing the DMRG algorithm in Python with detailed explanations and visualizations.
- **dmrg.jl**: A Julia implementation of the DMRG algorithm for high-performance computations.
- **python_code_without_dmrg.py**: Auxiliary Python code for quantum systems without explicit DMRG implementation.
- **README.md**: Documentation for the project.

---

## **Features**
- Implementation of the DMRG algorithm for 1D quantum systems.
- Support for spin chain models (e.g., Heisenberg model).
- Visualization of entanglement entropy and energy convergence.
- Modular code structure for easy extension to other models.

---

## **Requirements**
### Python Environment:
- Python 3.8+
- Required packages:
  - `numpy`
  - `matplotlib`
  - `scipy`
  
### Julia Environment:
- Julia 1.9+
- Required packages:
  - `LinearAlgebra`
  - `Plots`

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/DMRG-project.git
   cd DMRG-project
   ```
2. Install dependencies:
   - For Python:
     ```bash
     pip install numpy matplotlib scipy
     ```
   - For Julia:
     ```julia
     using Pkg
     Pkg.add("LinearAlgebra")
     Pkg.add("Plots")
     ```

---

## **Usage**
### Python Implementation:
Run the Jupyter Notebook `DMRG_python.ipynb` to explore step-by-step explanations and results:
```bash
jupyter notebook DMRG_python.ipynb
```

### Julia Implementation:
Execute the `dmrg.jl` file directly in Julia to run the DMRG algorithm on predefined models:
```bash
julia dmrg.jl
```

### Auxiliary Code:
Use `python_code_without_dmrg.py` for additional computations or as a starting point for custom implementations.

---

## **Examples**
1. **Heisenberg Spin Chain**:
   - Compute ground state energy using DMRG.
   - Visualize entanglement entropy across iterations.

2. **Custom Hamiltonians**:
   - Modify the Hamiltonian in either Python or Julia scripts to study other quantum systems.

---

## **Acknowledgments**
This project is inspired by pedagogical resources on DMRG and tensor networks, including works by Steven White and Ulrich Schollwöck.

--- 
