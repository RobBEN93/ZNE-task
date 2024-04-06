## ZNE task
This is a simple code demonstrating the ZNE (zero noise extrapolation) technique for error mitigation. In this case I went for simplicity with simple circuits and basic single-qubit and two-qubit depolarizing noise models.

## Key Components
- **Qiskit Aer Simulator**: Utilized for simulating both ideal and noisy quantum circuits, serving as the backbone for executing quantum algorithms in a controlled simulation environment.
- **Aer's NoiseModel Class**: Used for composing the depolarizing noise model in single and two-qubit gate operations.
- **Circuit Generation**: Generates random quantum circuits tailored to specified qubit numbers and depths, incorporating only gates from a specific set of single and two-qubit gates for simplicity and easy implementation of the noise model. Then an "identity circuit" is constructed by appending the inverse of the randomly selected gates for straightforward benchmarking.
- **Unitary Folding and Noise Scaling**: Introduces artificial noise scaling by applying unitary folding—repeating each gate followed by its inverse multiple times within the quantum circuit. This process is vital for preparing circuits with scaled levels of noise, essential for the ZNE methodology.
- **Estimator Methods**: A pivotal feature in Qiskit Aer for efficiently calculating expectation values of observables from quantum circuits under both ideal and noisy conditions in a very straightforward manner
- **Extrapolation**: Applies linear, polynomial, and exponential extrapolation techniques to estimate the circuit's behavior in the hypothetical zero-noise limit. These methods are implemented using NumPy's `polyfit` for linear and polynomial fits and SciPy's `curve_fit` for the exponential model, based on the noise-scaled expectation values.

## Observables
Focuses on measuring the observable `Z^n` (where `n` is the number of qubits), chosen for its relevance in assessing quantum state alignment with the computational basis, a common measure in quantum computing experiments.

## Simulation and Results
- Demonstrates the potential of ZNE for significantly mitigating noise effects, highlighted through the calculated expectation values and the observed improvements in the extrapolated, noise-mitigated results.

## Conclusion
This project leverages Qiskit Aer's advanced simulation capabilities, including the `NoiseModel` class and `Estimator` methods, to explore and validate ZNE as a viable noise mitigation strategy in quantum computing.

The used circuit generator and noise model are limited in scope but the local unitary folding function, noise scaling and extrapolation techniques can work on any circuit with standard gates (for which the .inverse() method can be applied).

Overall it was a fun project and a very interesting first hands-on approach to qiskit. :)

## Setup and Installation

Ensure you have the following packages installed to run the project:

- Python 3.7 or newer
- `qiskit>=1.0`
- `qiskit-aer`
- `numpy`
- `scipy`

You can install these packages using pip by running:

```bash
pip install qiskit qiskit-aer numpy scipy
