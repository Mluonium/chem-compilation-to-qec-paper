import mpmath
from pygridsynth import gridsynth_circuit
from pygridsynth.quantum_gate import HGate, TGate, SGate, SXGate, WGate
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import hadamard_transform
import qutip as q
import numpy as np

X = q.sigmax()
Z = q.sigmaz()
I = q.qeye(2)

c1 = 0.78796736
c2 = 0.18128881

t = np.pi / (c1 + c2)
theta = t / 2

def rz(theta):
    return (-1j * theta * Z / 2).expm()
def cnot():
    P0 = (I + Z) / 2
    P1 = (I - Z) / 2
    return (q.tensor(P0, I) + q.tensor(P1, X))

CNOT = cnot()
RZ = q.tensor(I, rz(theta))
H = hadamard_transform(1)

ZZ_decomposed = CNOT * RZ * CNOT
ZX_decomposed = (q.tensor(I, H) * CNOT * RZ * CNOT * q.tensor(I, H))

def generate_qpe_circuit():
    qc = QubitCircuit(4)

    qc.add_gate("H", targets=0)
    qc.add_gate("H", targets=1)
    qc.add_gate("H", targets=2)
    qc.add_gate("X", targets=3)

    for i in range(3):
        for _ in range(2**i):
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("RZ", targets=3, arg_value=theta)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("H", targets=3)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("RZ", targets=3, arg_value=theta)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("H", targets=3)
            qc.add_gate("H", targets=3)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("RZ", targets=3, arg_value=theta)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("H", targets=3)
            qc.add_gate("CNOT", controls=i, targets=3)
            qc.add_gate("RZ", targets=3, arg_value=theta)
            qc.add_gate("CNOT", controls=i, targets=3)

    qc.add_gate("SWAP", targets=[0, 2])
    qc.add_gate("H", targets=2)
    qc.add_gate("CPHASE", targets=1, controls=2, arg_value=-np.pi / 2)
    qc.add_gate("H", targets=1)
    qc.add_gate("CPHASE", targets=0, controls=2, arg_value=-np.pi / 4)
    qc.add_gate("CPHASE", targets=0, controls=1, arg_value=-np.pi / 2)
    qc.add_gate("H", targets=0)

    return qc


def add_gridsynth_rz(qc,target,theta,bits):
    epsilon=mpmath.mpf(2)**(-bits)
    circuit=gridsynth_circuit(theta=theta,epsilon=epsilon,wires=[0])
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


def add_cphase_decomposition(qc,control,target,theta,bits):
    add_gridsynth_rz(qc,control,theta/2,bits)
    add_gridsynth_rz(qc,target,theta/2,bits)
    qc.add_gate("CNOT",controls=control,targets=target)
    add_gridsynth_rz(qc,target,-theta/2,bits)
    qc.add_gate("CNOT",controls=control,targets=target)

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