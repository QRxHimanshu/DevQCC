from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import *
from qiskit_aer import AerSimulator, noise
from qiskit_ibm_runtime import QiskitRuntimeService
from scipy.optimize import linear_sum_assignment
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeProvider
from qiskit_aer.noise import NoiseModel

def get_gate_error(properties, gate_name, qubit_indices):
    for gate in properties.gates:
        if gate.gate == gate_name and gate.qubits == qubit_indices:
            for param in gate.parameters:
                if param.name == 'gate_error':
                    return param.value
    raise ValueError(f"Gate error not found for {gate_name} on qubits {qubit_indices}")

def find_distributed_backend_matches(subcircuits, backends):
    alignment_scores = np.zeros((len(subcircuits), len(backends)))
    
    for j, bname in enumerate(backends):
        provider = FakeProvider()
        backend = provider.get_backend(name=bname)
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        simulator = AerSimulator(noise_model=noise_model, coupling_map=backend.configuration().coupling_map, basis_gates=noise_model.basis_gates)

        for i, circuit in enumerate(subcircuits):
            # transpiled_circuit = transpile(circuit, backend=simulator)
            transpiled_circuit = transpile(circuit, backend=backend)

            properties = backend.properties()
            total_error = 1

            # Calculate total gate errors
            for gate in transpiled_circuit.data:
                gate_name = gate[0].name
                qubit_indices = [q._index for q in gate[1]]
                try:
                    gate_error = get_gate_error(properties, gate_name, qubit_indices)
                    total_error *= (1-gate_error)
                except ValueError as e:
                    print(e)
                    continue
            time_param = 0
            # Consider decoherence: Use T1, T2 values
            # This part is simplified and needs specific calculations based on circuit timing and coherence times
            for qubit in transpiled_circuit.qubits:
                qubit_index = qubit._index
                t1 = properties.t1(qubit_index)
                t2 = properties.t2(qubit_index)
                time_param += 1 - (t1 * t2) # Simplified example

            # Readout errors
            for bit in transpiled_circuit.clbits:
                qubit_index = bit._index
                readout_error = properties.readout_error(qubit_index)
                total_error *= (1-readout_error)

            alignment_scores[i][j] = total_error
            # alignment_scores[i][j] = max(0, min(score, 1))
    
    subcircuit_indices, backend_indices = linear_sum_assignment(-alignment_scores)
    
    # Calculate the total sum value of the optimal assignment
    total_optimal_value = alignment_scores[subcircuit_indices, backend_indices].sum()

    # Create a list where the index is subcircuit_idx and the value is its corresponding backend_idx
    optimal_assignments_list = [-1] * len(subcircuits)  # Initialize with -1 or a default value
    for subcircuit_idx, backend_idx in zip(subcircuit_indices, backend_indices):
        optimal_assignments_list[subcircuit_idx] = backend_idx
    
    return total_optimal_value, optimal_assignments_list

def find_optimal_backend_matches(subcircuits, backends):
    alignment_scores = np.zeros((len(subcircuits), len(backends)))
    
    for j, bname in enumerate(backends):
        provider = FakeProvider()
        backend = provider.get_backend(name=bname)
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        simulator = AerSimulator(noise_model=noise_model, coupling_map=backend.configuration().coupling_map, basis_gates=noise_model.basis_gates)

        for i, circuit in enumerate(subcircuits):
            transpiled_circuit = transpile(circuit, backend=backend)
            properties = backend.properties()
            total_error = 1

            # Calculate total gate errors
            for gate in transpiled_circuit.data:
                gate_name = gate[0].name
                qubit_indices = [q._index for q in gate[1]]
                try:
                    gate_error = get_gate_error(properties, gate_name, qubit_indices)
                    total_error *= (1-gate_error)
                except ValueError as e:
                    print(e)
                    continue

            # Consider decoherence: Use T1, T2 values
            # This part is simplified and needs specific calculations based on circuit timing and coherence times
            time_param = 0
            for qubit in transpiled_circuit.qubits:
                qubit_index = qubit._index
                t1 = properties.t1(qubit_index)
                t2 = properties.t2(qubit_index)
                time_param += 1 - (t1 * t2) # Simplified example

            # Readout errors
            for bit in transpiled_circuit.clbits:
                qubit_index = bit._index
                readout_error = properties.readout_error(qubit_index)
                total_error *= (1-readout_error)

            alignment_scores[i][j] = total_error
    
    optimal_assignments_list = []
    for i in range(len(subcircuits)):
        best_backend_idx = np.argmax(alignment_scores[i])
        optimal_assignments_list.append(best_backend_idx)

    total_optimal_value = alignment_scores[np.arange(len(subcircuits)), optimal_assignments_list].sum()
    
    return total_optimal_value, optimal_assignments_list

def find_worst_backend_matches(subcircuits, backends):
    alignment_scores = np.zeros((len(subcircuits), len(backends)))
    
    for j, bname in enumerate(backends):
        provider = FakeProvider()
        backend = provider.get_backend(name=bname)
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        simulator = AerSimulator(noise_model=noise_model, coupling_map=backend.configuration().coupling_map, basis_gates=noise_model.basis_gates)

        for i, circuit in enumerate(subcircuits):
            transpiled_circuit = transpile(circuit, backend=backend)
            properties = backend.properties()
            total_error = 1

            # Calculate total gate errors
            for gate in transpiled_circuit.data:
                gate_name = gate[0].name
                qubit_indices = [q._index for q in gate[1]]
                try:
                    gate_error = get_gate_error(properties, gate_name, qubit_indices)
                    total_error *= (1-gate_error)
                except ValueError as e:
                    print(e)
                    continue

            # Consider decoherence: Use T1, T2 values
            # This part is simplified and needs specific calculations based on circuit timing and coherence times
            time_param = 0
            for qubit in transpiled_circuit.qubits:
                qubit_index = qubit._index
                t1 = properties.t1(qubit_index)
                t2 = properties.t2(qubit_index)
                time_param += 1 - (t1 * t2) # Simplified example

            # Readout errors
            for bit in transpiled_circuit.clbits:
                qubit_index = bit._index
                readout_error = properties.readout_error(qubit_index)
                total_error *= (1-readout_error)

            alignment_scores[i][j] = total_error
    
    optimal_assignments_list = []
    for i in range(len(subcircuits)):
        best_backend_idx = np.argmin(alignment_scores[i])
        optimal_assignments_list.append(best_backend_idx)

    total_optimal_value = alignment_scores[np.arange(len(subcircuits)), optimal_assignments_list].sum()
    
    return total_optimal_value, optimal_assignments_list

# Example usage remains the same; this function now returns a dictionary for direct use.

# Example Usage
if __name__ == "__main__":
    # Define your backends (real or fake) and subcircuits here
    service = QiskitRuntimeService(channel='ibm_quantum')
    backends= []
    backend_list = service.backends()
    for backend in backend_list:
        # Skip simulators
        if backend.configuration().simulator:
            continue
        else :
            backends.append(backend)
    # backends = [FakeBoeblingen(), FakeVigo(), FakeYorktown()]
    subcircuits = [QuantumCircuit(2) for _ in range(3)]  # Placeholder for actual subcircuits
    
    # Populate subcircuits with some operations (Example)
    for i in range(len(subcircuits)):
        subcircuits[i].h(0)
        subcircuits[i].cx(0, 1)
        subcircuits[i].measure_all()
    
    optimal_matches = find_optimal_backend_matches(subcircuits, backends)
    
    # Print the optimal matches
    for subcircuit_idx, backend_idx in optimal_matches:
        print(f"Subcircuit {subcircuit_idx} is best matched with Backend {backends[backend_idx].name}.")
