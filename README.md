Example notebooks.
  
Run_Ethylene-4-qubit.ipynb is a quantum computing run on a simulator where the integrals are precomputed.
For this notbook those integrals are in data, but it is too big to put in github (max 100 MB) . 
The projectq simulator was used here, we can switch this to qiskit.

Downloaded Li2 through OpenFermion - Latest.ipynb does a couple of things, but has a couple of examples
of how we generate the integrals from Psi4, which can then be fed into the quantum computing simulator

vqse.py shows how we connect with the IBM hardware

The two components need to be integrated together and cleaned up.


Python libraries needed:

pip install qiskit
pip install openfermion
pip install projectq
pip install openfermionpsi4

OpenFermion has a docker container. All we need to do is add qiskit.

We can also do https://www.qiskit.org/aqua, which contains https://www.qiskit.org/documentation/apidoc/chemistry/chemistry.html .

https://github.com/Qiskit/qiskit-aqua/blob/master/README.md#running-a-chemistry-experiment .

