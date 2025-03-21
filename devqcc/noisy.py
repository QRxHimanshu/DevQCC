from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import Fake27QPulseV1
import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_alignment_score(circuit, backend):
    # Build noise model from backend properties
    noise_model = NoiseModel.from_backend(backend)

    # Get coupling map from backend
    coupling_map = backend.configuration().coupling_map

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    # Perform a noise simulation
    simulator = AerSimulator(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
    transpiled_circuit = transpile(circuit, simulator)

    # Count two-qubit gates and how many are directly supported by the backend's coupling map
    two_qubit_gates = 0
    aligned_gates = 0
    for inst, qubits, _ in transpiled_circuit.data:
        if len(qubits) == 2:
            two_qubit_gates += 1
            # Check if the qubits match any pair in the coupling map
            if qubits[0]._index < qubits[1]._index:
                qubit_pair = [qubits[0]._index, qubits[1]._index]
            else:
                qubit_pair = [qubits[1]._index, qubits[0]._index]
                
            if qubit_pair in coupling_map:
                aligned_gates += 1

    # Calculate alignment score as the proportion of aligned gates
    alignment_score = 1  # Default to perfect alignment if no two-qubit gates are present
    if two_qubit_gates > 0:
        alignment_score = aligned_gates / two_qubit_gates
    # depth = transpiled_circuit.depth()

    return alignment_score

def find_best_backend_matches(subcircuits, backends):
    """
    Finds the best match between a list of subcircuits and a list of quantum computing backends
    based on the alignment score.
    
    Parameters:
    - subcircuits: A list of QuantumCircuit objects.
    - backends: A list of Backend objects.
    
    Returns:
    A list of tuples, each containing the index of a subcircuit and the index of a backend that forms the best match.
    """
    num_subcircuits = len(subcircuits)
    num_backends = len(backends)
    
    # Initialize a matrix to store the alignment scores for each subcircuit-backend pair
    alignment_scores = np.zeros((num_subcircuits, num_backends))
    
    # Calculate alignment scores
    for i, subcircuit in enumerate(subcircuits):
        for j, backend in enumerate(backends):
            alignment_scores[i, j] = calculate_alignment_score(subcircuit, backend)
    
    # Convert the alignment scores to a cost matrix. The Hungarian algorithm minimizes cost,
    # so we subtract the scores from 1 (assuming alignment scores are normalized between 0 and 1).
    cost_matrix = 1 - alignment_scores
    
    # Apply the Hungarian algorithm to find the optimal assignment of subcircuits to backends
    subcircuit_indices, backend_indices = linear_sum_assignment(cost_matrix)
    
    # Prepare and return the list of optimal matches
    optimal_matches = list(zip(subcircuit_indices, backend_indices))
    return optimal_matches


def calculate_additional_swap_gates(circuit, backend):
    # Use a basic swap pass to add swaps for a linear topology
    pass_manager = PassManager(BasicSwap(coupling_map=backend.configuration().coupling_map))
    swapped_circuit = pass_manager.run(circuit)
    
    # Transpile the circuit for the backend with optimization level 0 to minimize changes
    transpiled_circuit = transpile(swapped_circuit, backend, optimization_level=0)
    
    # Calculate the difference in gate count
    initial_gate_count = sum(circuit.count_ops().values())
    final_gate_count = sum(transpiled_circuit.count_ops().values())
    
    additional_swaps = final_gate_count - initial_gate_count
    return additional_swaps

def calculate_circuit_depth(circuit, backend):
    # Transpile the circuit for the backend to account for its architecture
    transpiled_circuit = transpile(circuit, backend)
    
    # Get the circuit depth
    depth = transpiled_circuit.depth()
    return depth

def calculate_gate_fidelity(circuit, backend):
    # Get the properties of the backend
    properties = backend.properties()
    
    # Get the operation counts in the circuit
    op_counts = circuit.count_ops()
    
    total_fidelity = 0
    total_ops = 0
    
    for op, count in op_counts.items():
        # Lookup the gate fidelity from the backend properties
        if op in properties.gate_error:
            fidelity = 1 - properties.gate_error(op)
            total_fidelity += fidelity * count
            total_ops += count
    
    # Calculate the average fidelity weighted by operation count
    if total_ops > 0:
        average_fidelity = total_fidelity / total_ops
    else:
        average_fidelity = 0
    
    return average_fidelity


# Example usage would now look like this:
# (assuming `backend` is a Backend object you have from either a real or a fake provider)
# circ = QuantumCircuit(5, 5)
# circ.h(0)
# circ.cx(0, 1)
# circ.cx(1, 2)
# circ.cx(2, 3)
# circ.cx(3, 4)
# circ.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
# alignment_score = calculate_alignment_score(circ, backend)
# print(alignment_score)
