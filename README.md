# Reproduce: Quantum chemistry compilation to QEC
This repository contains code for reproducing some parts of the paper [Compilation of a simple chemistry application to quantum error correction primitives](https://arxiv.org/pdf/2307.03233). This paper is a simple chemistry problem, specifically finding the ground state energy of dihydrogen using Quantum Phase-Estimation (QPE), and compiling this problem to Quantum Error Correction (QEC) methods.

Our repository contains four main parts:
1. Constructing the Hamiltonian
2. Trotterization
3. Gridsynth software package
4. Resource estimation

Each part has it's own `jupyter` file. Further, there is a helper file for the QPE circuits and (de)composition `generate_qpe.py`.

### Constructing the Hamiltonian: `hamiltonian.ipynb`

Contains a stepwise guide on constructing the hamiltonian. The steps can be roughly described as:
- Molecular geometry
- STO-3G and LCAO
- One- and two- electron integrals
- Second-quantized Hamiltonian
- Jordan-Wigner
- 4-qubit Pauli Hamiltonian
- Tapering

### Trotterization: `trotterization.ipynb`

To simulate a Hamiltonian, we need to be able to implement the time evolution operator for a time $t$, which is given by the unitary operator $U(t) = e^{-iH}$. Trotterization allows us to approximate this using the native get set only. Our notebook goes through first- and second-order Trotterization. We then use QPE to estimate the ground state energy of the Hamiltonian.


### Gridsynth: `gridsynth.ipynb`

Gridsynth is an important part of the paper and is used to approximate a single qubit rotation around the Z-axis circuit using the Cliford+T gate set. We use the Python version [`pygridsynth`](https://github.com/quantum-programming/pygridsynth).

The main tasks completed in the notebook are as follows:
- Circuit generation: create QPE circuit → decompose using Gridsynth
- Import the helper functions to handle the gates and make a QPE simulation function 
- Look at total variation distance (between original QPE and decomposed QPE) and gate count

This is the main notebook that uses the helper file, as it uses every function.

### Resource estimation: `resource_estimation.ipynb`

The main goal of this notebook is to restimate the resources needed for fault tolerant QPE on the simple chemistry problem. We talk about:
- Error rates
- Noise model
- QEC rounds per gate

This leads three to plots, that compare code distance with one metric each. The metrics are:
- Logical error rate
- Total QEC measurement rounds
- Physical qubit footprint
