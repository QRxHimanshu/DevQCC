import os, math
import os, logging
from qiskit import *

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Comment this line if using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from devqcc_runtime.main import devqcc # Use this just to benchmark the runtime

from devqcc.main import devqcc # Use this for exact computation

from helper_functions.benchmarks import generate_circ
from helper_functions.metrics import chi2_distance
from helper_functions.metrics import fidelity
from helper_functions.non_ibmq_functions import evaluate_circ

if __name__ == "__main__":
    # lst = [6,8,10,12]
    circs = {"bv":[26],"supremacy":[25],"adder":[24],"hwea":[24]}
    for circ_type,lst in circs.items():
        for circ_size in lst:
            print("NEW CIRCUIT :",circ_type)
            print()
            print()
            circuit_type = circ_type
            circuit_size = circ_size
            circ = generate_circ(
                num_qubits=circuit_size,
                depth=1,
                circuit_type=circuit_type,
                reg_name="q",
                connected_only=True,
                seed=None,
            )

            devqcc = devqcc(
                
                name="%s_%d" % (circuit_type, circuit_size),
                circuit=circ,
                cutter_constraints={
                    "max_subcircuit_width": min(circ_size,5),
                    "max_subcircuit_cuts": 20,
                    "subcircuit_size_imbalance": 3,
                    "max_cuts": 20,
                    "num_subcircuits": [2, 3, 4, 5, 6, 7],
                },
                # cutter_constraints={
                #     "max_subcircuit_width": math.ceil(circuit.num_qubits / 4 * 3),
                #     "max_subcircuit_cuts": 20,
                #     "subcircuit_size_imbalance": 10,
                #     "max_cuts": 25,
                #     "num_subcircuits": [2, 3, 4, 5, 6],
                # },
                verbose=True,
            )
            devqcc.cut()
            sv = evaluate_circ(circuit=circ,backend='statevector_simulator')
            if not devqcc.has_solution:
                raise Exception("The input circuit and constraints have no viable cuts")

            devqcc.evaluate(eval_mode="noisy", num_shots_fn=None)
            devqcc.build(mem_limit=32, recursion_depth=1)
            print("Cut: %d recursions." % (devqcc.num_recursions))
            devqcc.verify()
            print("Mse :",devqcc.approximation_error)
            print("Chi_square : ",devqcc.chi2_dist)
            # print("Fidelity : ",devqcc.fidel)
            print("Hellinger : ", devqcc.hellfi)
            print("Relative Entropy : ",devqcc.relent)
            # print("KLDivergence : ",devqcc.klfid)
            # print("Jsfid : ", devqcc.jsfid)
            print("Tvdfid : ",devqcc.tvdf)
            print("Jsfid : ",devqcc.jsfid)
            print("Quartile 1 HOP : ", devqcc.quar1)
            print("Quartile 2 HOP : ", devqcc.quar2)
            print("Quartile 3 HOP : ", devqcc.quar3)
            print(devqcc.times)
            tot_time = 0
            for key,value in devqcc.times.items():
                tot_time += value
            print("Total simulation time: ",tot_time)
            devqcc.clean_data()
print("hi")