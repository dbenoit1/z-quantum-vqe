################################################################################
# © Copyright 2021 Zapata Computing Inc.
# DMB mods to 0 - CNOT - 0 04/05/2022
################################################################################
from typing import List, Optional 

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import X,CNOT, RY, Circuit
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import ansatz_property

#from .utils import build_hartree_fock_circuit

class HF_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")

    def __init__(self, number_of_layers: int, number_of_qubits: int, nb_occ: int):
        """An ansatz implementation for the Hardware Efficient Quantum Compiling Ansatz
            used in https://arxiv.org/pdf/2011.12245.pdf
            modified to be only 0 - CNOT - 0

        Args:
            number_of_layers: number of layers in the circuit.
            number_of_qubits: number of qubits in the circuit.
            nb_occ: number of occupied states (spin orbitals, for example)

        Attributes:
            number_of_qubits: See Args
            number_of_layers: See Args
        """
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be a positive integer")
        super().__init__(number_of_layers)
        assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ

    def _build_rotational_subcircuit(
        self, circuit: Circuit, parameters: np.ndarray
    ) -> Circuit:
        """Add the subcircuit which includes several rotation gates acting on each qubit

        Args:
            circuit: The circuit to append to
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            circuit with added rotational sub-layer
        """
        # Add RY(theta)
        zero=1.0
        for qubit_index in range(self.number_of_qubits):

            qubit_parameters = parameters[qubit_index : (qubit_index + 1)]
            circuit += RY(qubit_parameters[0]*zero)(qubit_index)
            #circuit += RY(qubit_parameters[0])(qubit_index)
            #this part doesnt work here
            #circuit += RY(-qubit_parameters[0])(qubit_index)

        return circuit
    
    def _build_occ_cnot(
        self, circuit: Circuit, a:int, b:int
    ) -> Circuit:
        """Add a subcircuit that performs a cnot for occupied states X(a)-cnot(a,b)-X(a)

        Args:
            circuit: The circuit to append to
            a,b: qubit number 

        Returns:
            circuit with added CNOT
        """
        # Add X-CNOT-X
        circuit+=X(a)
        circuit += CNOT(a, b)
        circuit+=X(a)

        return circuit

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        """Build circuit layer for the hardware efficient quantum compiling ansatz

        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Quantum
            Compiling Ansatz
        """
        circuit_layer = Circuit()
        
 #       circuit_layer += build_hartree_fock_circuit(
 #           self.number_of_qubits,
 #           self.nb_occ/2,
 #           self.nb_occ/2,
 #           "Jordan-Wigner",
 #           #self.number_of_alpha_electrons,
 #           #self._number_of_beta_electrons,
 #           #self._transformation,
 #       )

        # Hardwired JW HF ansatz
        for i in range(self.nb_occ):
            #adds a not (i.e. |1>) for each occupied state starting from 0 up to nb_occ (-1 coz python.. )
            circuit_layer += X(i)

        # Add RY(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer, parameters[: self.number_of_qubits]
        )

        qubit_ids = list(range(self.number_of_qubits))
        # Add  CNOT(x, x+1) for x in all (qubits)
      #  for control, target in zip(
      #      qubit_ids[:-2:], qubit_ids[1::]
      #  ):  # loop over qubits 0, 1, 2, 3,...
      ##      if ((self.nb_occ >0) and ((control == 0) or (control == self.number_of_qubits/self.nb_occ) )):
      ##          circuit_layer = self._build_occ_cnot(circuit_layer, control,target)
      ##      else:
      ##          circuit_layer += CNOT(control, target)
      #     circuit_layer += CNOT(control, target)

        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: 2011.12245, Fig. 1)

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
