# DevQCC
The DevQCC codebase is useful for those working in:
* Distributed Quantum Computing
* Quantum Machine Learning (QML)

## Working with the github repository
DevQCC is the backend code for the paper DevQCC: Device-Aware Quantum Circuit Cutting Framework with Applications in Quantum Machine Learning. The codes are adopted from CutQC [1] which  cuts large quantum circuits into smaller subcircuits and runs on small quantum computers. We identified that there is a research gap in the proper framework which is device aware so we appended the CutQC framework to append device awareness and found out that device aware circuit cutting and distribution is more effective in terms of fidelity and accuracy in QML applications. We also extended the DevQCC framework for machine learning applications as its requirements are quite different from other quantum circuits due to the data intensive and iterative nature.   The code base is useful for those who work in distributed quantum computing as well as quantum machine learning. 


## Important note
Kindly note that the solution is very compute intensive for circuits beyond 20 qubits. The noisy simulation requires high computations so it is recommended to perform on a high end computer since it might result in crashing the system due to memory exhaustion. The results are simulation based so the real behavior might differ but the code provides significant validation of the proposed hypothesis that the device aware quantum circuit is more effective that naive cutting or techniques designed only to minimize the classical computations.  


## Installation
Make a Python 3.10 or above  virtual environment:
DevQC uses the Gurobi solver. Obtain and install a Gurobi license. Follow the instructions.
Install required packages:
pip install -r requirements.txt
Example Code
For an example, run:

## Citing DevQCC
If you use DevQCC in your work, we would appreciate it if you cite our paper:

Sahu, Himanshu, Gupta, Hari Prabhat,Vardhan, Vishnu Puvvada,Mishra Rahul. "DevQCC: Device-Aware Quantum Circuit Cutting Framework with Applications in Quantum Machine Learning." Quantum Mach. Intell. X, XX (2025). https://doi.org/10.XXXXXXXXXXXXX

## Code Contributor 
1. Vishnu Vardhan Puvvada
2. Dr. Rahul Mishra

Contact Us
* Please open an issue on this repository for support.
* Please reach out to Himanshu Sahu.

[1]. Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

