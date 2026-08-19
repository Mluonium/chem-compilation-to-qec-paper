import mpmath
from pygridsynth import gridsynth_circuit
from pygridsynth.quantum_gate import HGate, TGate, SGate, SXGate, WGate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import hadamard_transform
import qutip as q
import numpy as np

c1 = 0.78796736
c2 = 0.18128881

t = np.pi / (c1 + c2)
theta = t / 2

# --------------------------------------------------
# MAIN QPE CIRCUIT GENERATOR
# --------------------------------------------------
def generate_qpe_circuit():
    qc = QubitCircuit(4)

    # phase-estimation qubits are put into an equal superposition
    qc.add_gate("H", targets=0)
    qc.add_gate("H", targets=1)
    qc.add_gate("H", targets=2)

    # eigenstate qubit
    qc.add_gate("X", targets=3)

    # perform controlled-unitary operation for each of the 3 PE qubits
    for i in range(3):
        power = 2 ** i # unitary power, creates: U^1, U^2, U^4
        angle = power * theta # power multiplied by angle: θ, 2θ, 4θ 

        # first controlled interaction: ZZ
        qc.add_gate("CNOT", controls=i, targets=3)
        qc.add_gate("RZ", targets=3, arg_value=angle)
        qc.add_gate("CNOT", controls=i, targets=3)

        # Hadamard on target to change basis: Z -> X
        qc.add_gate("H", targets=3)

        # second controlled interaction: ZX
        qc.add_gate("CNOT", controls=i, targets=3)
        qc.add_gate("RZ", targets=3, arg_value=angle)
        qc.add_gate("CNOT", controls=i, targets=3)

        # third controlled interaction: ZX
        qc.add_gate("CNOT", controls=i, targets=3)
        qc.add_gate("RZ", targets=3, arg_value=angle)
        qc.add_gate("CNOT", controls=i, targets=3)

        # change basis again: X -> Z
        qc.add_gate("H", targets=3)

        # fourth controlled interaction: ZZ
        qc.add_gate("CNOT", controls=i, targets=3)
        qc.add_gate("RZ", targets=3, arg_value=angle)
        qc.add_gate("CNOT", controls=i, targets=3)

    # inverse QFT
    qc.add_gate("SWAP", targets=[0, 2])
    qc.add_gate("H", targets=2)
    qc.add_gate("CPHASE", controls=2, targets=1, arg_value=-np.pi / 2)
    qc.add_gate("H", targets=1)
    qc.add_gate("CPHASE", controls=2, targets=0, arg_value=-np.pi / 4)
    qc.add_gate("CPHASE", controls=1, targets=0, arg_value=-np.pi / 2)
    qc.add_gate("H", targets=0)

    return qc

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

# Implement the main part of gridsynth, to take an angle and approximate it using cliffords 
def add_gridsynth_rz(qc,target,theta,bits):
    epsilon=mpmath.mpf(2)**(-bits) # approximation error
    circuit=gridsynth_circuit(theta=theta, epsilon=epsilon, wires=[0], up_to_phase=True) # actual gridsynth circuit

    # go through the gates and add them to the qutip circuit 
    for gate in reversed(circuit):
        if isinstance(gate,HGate):
            qc.add_gate("H",targets=target)
        elif isinstance(gate,TGate):
            qc.add_gate("T",targets=target)
        elif isinstance(gate,SGate):
            qc.add_gate("S",targets=target)
        elif isinstance(gate,SXGate):
            qc.add_gate("X",targets=target)
        elif isinstance(gate,WGate):
            continue
        else:
            raise TypeError(f"Unknown Gridsynth gate: {type(gate).__name__}")

# decomposing CPHASE into simpler gates
def add_cphase_decomposition(qc,control,target,theta,bits):
    add_gridsynth_rz(qc,control,theta/2,bits)
    add_gridsynth_rz(qc,target,theta/2,bits)
    qc.add_gate("CNOT",controls=control,targets=target)
    add_gridsynth_rz(qc,target,-theta/2,bits)
    qc.add_gate("CNOT",controls=control,targets=target)

# decomposing RZ and CPHASE with the previous helper functions
def decompose_qpe_with_gridsynth(qc,bits):
    new_qc=QubitCircuit(qc.N)
    for gate in qc.gates:
        if gate.name=="RZ":
            add_gridsynth_rz(new_qc,gate.targets[0],gate.arg_value,bits)
        elif gate.name=="CPHASE":
            add_cphase_decomposition(new_qc,gate.controls[0],gate.targets[0],gate.arg_value,bits)
        else:
            new_qc.add_gate(gate.name,targets=gate.targets,controls=gate.controls,arg_value=gate.arg_value)
    return new_qc

# implementing SWAP operator using Paulis
def swap_operator(a,b,N): # takes index of qubits to be swapped and total number of qubits
    X=q.sigmax()
    Y=q.sigmay()
    Z=q.sigmaz()
    I=q.qeye(2)
    ops=[]

    for pauli1,pauli2 in [(I,I),(X,X),(Y,Y),(Z,Z)]:
        op=[I for _ in range(N)] # create identity on every qubit
        op[a]=pauli1 # first pauli on qubit a
        op[b]=pauli2 # second on b
        ops.append(q.tensor(op)) # tensor all operators together
    return sum(ops)/2

# taking a gate 2-qubit gate U and embedding it in an N qubit system
def embed_two_qubit_gate(U,q1,q2,N):
    # create full U matrix
    dim=2**N
    full_U=np.zeros((dim,dim),dtype=complex)
    U=U.full()

    # loop through every possible input state
    for input_index in range(dim):
        input_bits=[(input_index>>(N-1-i))&1 for i in range(N)] # binary qubit values
        local_input=2*input_bits[q1]+input_bits[q2] # takes the 2 selected qubits and translates to 0-3 index

        # loop through all posible local outputs
        for local_output in range(4):
            output_bits=input_bits.copy()

            # select output qubits
            output_bits[q1]=local_output//2
            output_bits[q2]=local_output%2

            output_index=0
            for bit in output_bits:
                output_index=(output_index<<1)|bit # output bit string to integer index
            full_U[output_index,input_index]=U[local_output,local_input] # put matrix element into the bigger matrix
    return q.Qobj(full_U,dims=[[2]*N,[2]*N]) # return as qutip object

# samples measurement results from probability distribution and counts how many times each 4-bit outcome happened
def sample_counts(probabilities,shots=10000,rng=None):
    if rng is None:
        rng=np.random.default_rng()
    samples=rng.choice(len(probabilities),size=shots,p=probabilities)
    counts={}
    for index in samples:
        bitstring=format(index,"04b")
        counts[bitstring]=counts.get(bitstring,0)+1
    return counts