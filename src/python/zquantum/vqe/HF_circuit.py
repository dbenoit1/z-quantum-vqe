################################################################################
# © Copyright 2021 Zapata Computing Inc.
# DMB mods to [theta] - [-theta] - [CNOT] 04/05/2022
# TESTED WITH STATEVECTOR SIMS 06/05/2022 - works
# REQUIRES A BETTER HF INITIAL STATE - CURRENTLY ONLY WORKS FOR JW MAPPING 
# BUT IS HARDWIRED
# CURRENTLY GENERIC HF INITIALISATION THROWS AN ERROR
# cannot import name 'bravyi_kitaev' from 'zquantum.core.openfermion'
# SO REVERTING TO HARDWIRED OPTION
################################################################################
from typing import List, Optional 

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import X,CNOT, RY, Circuit
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import ansatz_property

# from .utils import build_hartree_fock_circuit

class HF_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")

    def __init__(self, number_of_layers: int, number_of_qubits: int, nb_occ: int):
        """An ansatz implementation for the Hardware Efficient Quantum Compiling Ansatz
            used in https://arxiv.org/pdf/2011.12245.pdf
            modified to be only Ry(+theta) => Ry(-Theta) => IdCNOT

        Args:
            number_of_layers: number of layers in the circuit.
            number_of_qubits: number of qubits in the circuit.
            nb_occ: number of occupied states (spin orbitals, for example)

        Attributes:
            number_of_qubits: See Args
            number_of_layers: See Args
            nb_occ: See Args
        """
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be a positive integer")
        super().__init__(number_of_layers)
       # assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ

    def _build_rotational_subcircuit(
        self, circuit: Circuit, parameters: np.ndarray
    ) -> Circuit:
        """Add the subcircuit which includes two rotation gates acting on each qubit

        Args:
            circuit: The circuit to append to
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            circuit with added rotational sub-layer
        """
        # Add RY(theta) RY(-theta)
        for qubit_index in range(self.number_of_qubits):

            qubit_parameters = parameters[qubit_index : (qubit_index + 1)]
            circuit += RY(qubit_parameters[0])(qubit_index)
            circuit += RY(-qubit_parameters[0])(qubit_index)

        return circuit
    
    def _build_not_cnot_not(
        self, circuit: Circuit, a:int, b:int
    ) -> Circuit:
        """Add a subcircuit that performs X(a)-cnot(a,b)-X(a) for occupied states 

        Args:
            circuit: The circuit to append to
            a,b: qubit number 

        Returns:
            circuit with added X(a)-cnot(a,b)-X(a)
        """
        # Add X-CNOT-X
        circuit+=X(a)
        circuit += CNOT(a, b)
        circuit+=X(a)

        return circuit

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        """Build circuit layer for the hardware efficient HF ansatz analog
 
        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Quantum
            Compiling Ansatz
        """
        circuit_layer = Circuit()
        
#        circuit_layer += build_hartree_fock_circuit(
#            number_of_qubits=self.number_of_qubits,
#            number_of_alpha_electrons=self.nb_occ/2,
#            number_of_beta_electrons=self.nb_occ/2,
#            transformation="Jordan-Wigner",
#        )

        # Hardwired JW HF ansatz instead (previous one has library issues)
        for i in range(self.nb_occ):
            #adds a not (i.e. |1>) for each occupied state starting from 0 up to nb_occ (-1 coz python.. )
            circuit_layer += X(i)

        # Add RY(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer, parameters[: self.number_of_qubits]
        )

        # Add CNOT(x, x+1) for x in all(qubits)
        #Slightly more hardwired approach:
        for i in range(self.number_of_qubits):
            target=i+1
            if (target<self.number_of_qubits):
                 if((self.nb_occ >0) and (i<self.nb_occ)):
                    #each occupied state starting from 0 up to nb_occ (-1 coz python.. ) is a X so use X-CNOT-X
                    circuit_layer = self._build_not_cnot_not(circuit_layer, i,i+1)
                 else:
                    circuit_layer += CNOT(i, i+1)
                
        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds an HF ansatz circuit

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = np.asarray(self.symbols, dtype=object)

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
        for layer_index in range(self.number_of_layers):
            circuit += self._build_circuit_layer(
                parameters[
                    layer_index
                    * self.number_of_params_per_layer : (layer_index + 1)
                    * self.number_of_params_per_layer
                ]
            )
        return circuit

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return self.number_of_params_per_layer * self.number_of_layers

    @property
    def number_of_params_per_layer(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return self.number_of_qubits 

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """
        Returns a list of symbolic parameters used for creating the ansatz.
        The order of the symbols should match the order in which parameters
        should be passed for creating executable circuit.
        """
        return [
            sympy.Symbol("theta_{}".format(i)) for i in range(self.number_of_params)
        ]
