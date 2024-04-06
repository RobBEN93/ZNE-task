# Task 3 ZNE
#
# Zero-noise extrapolation (ZNE) is a noise mitigation technique. It works by intentionally scaling the noise of a
# quantum circuit to then extrapolate the zero-noise limit of an observable of interest.
#
# In this task, you will build a simple ZNE function from scratch:
#
#    Build a simple noise model with depolarizing noise
#    Create different circuits to test your noise models and choose the observable to measure
#    Apply the unitary folding method.
#    Apply the extrapolation method to get the zero-noise limit.
#               Different extrapolation methods achieve different results, such as Linear, polynomial, and exponential.
#    Compare mitigated and unmitigated results

# Install qiskit, install qiskit-aer

import numpy as np
from qiskit.circuit import Gate
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
import random

# We define a couple of functions for generating random circuits from a specific gates for simplicity and also define
# a function for the unitary folding method

def custom_random_circuit(num_qubits, depth, single_qubit_gates, two_qubit_gates):
    # Initialize quantum circuit
    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        # Decide randomly between single and two qubit gates
        if num_qubits > 1 and random.choice([True, False]):
            gate = random.choice(two_qubit_gates)
            qubits = random.sample(range(num_qubits), 2)  # Pick two different qubits
            if gate == 'cx':
                qc.cx(qubits[0], qubits[1])
            elif gate == 'cz':
                qc.cz(qubits[0], qubits[1])
            elif gate == 'cy':
                qc.cy(qubits[0], qubits[1])
            elif gate == 'swap':
                qc.swap(qubits[0], qubits[1])
            elif gate == 'iswap':
                qc.iswap(qubits[0], qubits[1])
            elif gate == 'cu':
                # CU gate needs parameters, using default π/2 for simplicity
                qc.cu(np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, qubits[0], qubits[1])
            elif gate in ['crx', 'cry', 'crz']:
                # CR gates need an angle, using π/2 for simplicity
                angle = np.pi / 2
                if gate == 'crx':
                    qc.crx(angle, qubits[0], qubits[1])
                elif gate == 'cry':
                    qc.cry(angle, qubits[0], qubits[1])
                elif gate == 'crz':
                    qc.crz(angle, qubits[0], qubits[1])
        else:
            gate = random.choice(single_qubit_gates)
            qubit = random.choice(range(num_qubits))  # Pick one qubit
            # Applying single qubit gates
            if gate == 'p':
                qc.p(np.pi / 2, qubit)  # U1 needs an angle, using π/2 for simplicity
            elif gate == 'u':
                qc.u(np.pi / 2, 0, np.pi, qubit)  # U gate with parameters θ=π/2, φ=0, λ=π
            elif gate == 'id':
                qc.id(qubit)
            elif gate == 'x':
                qc.x(qubit)
            elif gate == 'y':
                qc.y(qubit)
            elif gate == 'z':
                qc.z(qubit)
            elif gate == 'h':
                qc.h(qubit)
            elif gate == 's':
                qc.s(qubit)
            elif gate == 'sdg':
                qc.sdg(qubit)
            elif gate == 't':
                qc.t(qubit)
            elif gate == 'tdg':
                qc.tdg(qubit)
    return qc
def create_identity_circuit(num_qubits, depth):
    # Create a random circuit of the given depth with no measurement operations
    circuit = custom_random_circuit(num_qubits, depth, single_qubit_gates, two_qubit_gates)

    # Create the inverse of this circuit
    inverse_circuit = circuit.inverse()

    # Append the inverse circuit to the original, creating an identity circuit
    identity_circuit = circuit.compose(inverse_circuit)

    return identity_circuit
def unitary_folding(circ, noise_scale):
    """
    This function performs a local unitary folding by taking the existing quantum circuit and returns a new circuit that
    includes all original instructions and additional 'n' (noise_scale) applications of each gate followed by its inverse for each
    original gate  for scaling up the noise.

    Parameters:
    circ (QuantumCircuit): The original quantum circuit to modify.
    noise_scale (int): The number of times to apply each gate and its inverse.

    Returns:
    locally folded QuantumCircuit: .
    """

    # Create a new quantum circuit with the same quantum and classical registers as the original
    new_circ = QuantumCircuit(*circ.qregs, *circ.cregs)

    for inst, qargs, cargs in circ.data:
        # Add the original instruction to the new circuit
        new_circ.append(inst, qargs, cargs)

        # Check if the instruction is a gate and has an inverse
        if isinstance(inst, Gate):
            # Get the inverse of the gate once before the loop
            inverse_gate = inst.inverse()

            for _ in range(noise_scale):
                # Add original gate and its inverse to the circuit 'n' times
                new_circ.append(inst, qargs, cargs)  # Same gate again
                new_circ.append(inverse_gate, qargs, cargs)  # Its inverse

    return new_circ

# DEFINING SIMPLE DEPOLARIZING ERROR NOISE MODEL AND APPLYING IT TO ALL GATES FROM OUR GENERATED CIRCUITS
# This example uses Qiskit's Aer simulator to model depolarizing noise

# Create an empty noise model from the NoiseModel class
noise_model = NoiseModel()

# Define probability of error
p_error=0.001

# Add depolarizing error to relevant single-qubit and two-qubit gates

single_qubit_depolar_error = pauli_error([
    ('I', 1 - p_error),
    ('X', p_error / 3),
    ('Y', p_error / 3),
    ('Z', p_error / 3)
])
single_qubit_gates=['p', 'u', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']

for gate in single_qubit_gates:
    noise_model.add_all_qubit_quantum_error(single_qubit_depolar_error, gate)

two_qubit_depolar_error = pauli_error([
    ('II', 1 - p_error),  # No error
    ('XX', p_error / 15),  # Depolarization errors
    ('YY', p_error / 15),
    ('ZZ', p_error / 15),
    ('IX', p_error / 15),
    ('XI', p_error / 15),
    ('IY', p_error / 15),
    ('YI', p_error / 15),
    ('IZ', p_error / 15),
    ('ZI', p_error / 15),
    ('XY', p_error / 15),
    ('YX', p_error / 15),
    ('XZ', p_error / 15),
    ('ZX', p_error / 15),
    ('YZ', p_error / 15),
    ('ZY', p_error / 15),
])
two_qubit_gates=['cx','cz','cy','swap','iswap','cu','crx','cry','crz']

for gate in two_qubit_gates:
    noise_model.add_all_qubit_quantum_error(two_qubit_depolar_error, gate)

# Print noise model info
print(noise_model)

# Create a random circuit equalling the identity for given number of qubits and given depth

num_qubits = 2
depth = 30
identity_circuit = create_identity_circuit(num_qubits, depth)

print(identity_circuit)

noise_scale=6

# Define the observable; for simplicity, we use a Z observable on each qubit
n_qbits = identity_circuit.num_qubits
observable = SparsePauliOp("Z" * n_qbits)

# Create Estimator instances for local ideal and noisy simulations
ideal_estimator = Estimator()
noisy_estimator = Estimator(backend_options={"noise_model": noise_model}, approximation=False)

# Run the estimators with your circuit and observable
ideal_result = ideal_estimator.run(identity_circuit, observables=[observable]).result()
noisy_result = noisy_estimator.run(identity_circuit, observables=[observable]).result()

# Extract and print the expectation value
ideal_expectation_value = ideal_result.values[0]
noisy_value = noisy_result.values[0]

print(f"Ideal value: {ideal_expectation_value}")
print(f"Noisy expectation value: {noisy_value}")

# Get expectation values for different noise scalings for extrapolation

noisy_values = []
noisy_values.append(noisy_value)

for n in range(noise_scale):
    # Modify your circuit with the current value of n
    ufolded_circuit = unitary_folding(identity_circuit,n)

    # Run the noisy simulation with the modified circuit
    noisy_result = noisy_estimator.run(ufolded_circuit, observables=[observable]).result()

    # Extract the expectation value and store it in your list
    noisy_value = noisy_result.values[0]
    noisy_values.append(noisy_value)
print()
print(f"Scaled noise expectation values for given scale {noise_scale}: {noisy_values}")
print()
# ERROR MITIGATION
print("Mitigated results:")
# PERFORM EXTRAPOLATION
x_zero_noise = 0
x = np.array(list(range(1, len(noisy_values) + 1)))
y = np.array(noisy_values)

# Linear model
linear_model = np.polyfit(x, y, 1)

# Linear extrapolation function
def linear_extrapolate(x_new):
    return linear_model[0] * x_new + linear_model[1]

# Linear ZNE
lin_zero_noise = linear_extrapolate(x_zero_noise)

print(f"Linear Extrapolation: <E({x_zero_noise})> = {lin_zero_noise:.5f}")

# Polynomial model (degree = 2)
polynomial_model = np.polyfit(x, y, 2)

# Extrapolation function
def polynomial_extrapolate(x_new):
    return polynomial_model[0] * x_new**2 + polynomial_model[1] * x_new + polynomial_model[2]

# Extrapolating a new value
pol_zero_noise = polynomial_extrapolate(x_zero_noise)
print(f"Polynomial Extrapolation: <E({x_zero_noise})> = {pol_zero_noise:.5f}")

# Define the exponential function
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit the exponential model
params, _ = curve_fit(exponential_func, x, y)

# Extrapolation function
def exponential_extrapolate(x_new):
    return exponential_func(x_new, *params)

# Extrapolating a new value
exp_zero_noise = exponential_extrapolate(x_zero_noise)
print(f"Exponential Extrapolation: <E({x_zero_noise})> = {exp_zero_noise:.5f}")

#COMPARISON

# Calculate absolute errors
error_linear = abs(lin_zero_noise - ideal_expectation_value)
error_polynomial = abs(pol_zero_noise - ideal_expectation_value)
error_exponential = abs(exp_zero_noise - ideal_expectation_value)

# Calculate improvement over noisy value
improvement_linear = abs(lin_zero_noise - noisy_value)
improvement_polynomial = abs(pol_zero_noise - noisy_value)
improvement_exponential = abs(exp_zero_noise - noisy_value)
print()
print("Error from ideal values:")
print(f"Linear Extrapolation Error: {error_linear:.5f}")
print(f"Polynomial Extrapolation Error: {error_polynomial:.5f}")
print(f"Exponential Extrapolation Error: {error_exponential:.5f}")
print()
print("Improvements over noisy:")
print(f"Linear: {improvement_linear:.5f}")
print(f"Polynomial: {improvement_polynomial:.5f}")
print(f"Exponential: {improvement_exponential:.5f}")
