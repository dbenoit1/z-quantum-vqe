################################################################################
# © Copyright 2021 Zapata Computing Inc.
# DMB mods to Ry - CNOT - Ry 04/05/2022
# 09/05/22 DMB: TESTED AND WORKS WITH STATEVECTOR & NFT
################################################################################
from typing import List, Optional

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import CNOT, RY, X, Circuit
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import ansatz_property


class HEA_RY_CNOT_RY_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")

    def __init__(self, number_of_layers: int, number_of_qubits: int, nb_occ: int):
        """An ansatz implementation for the Hardware Efficient Quantum Compiling Ansatz
            used in https://arxiv.org/pdf/2011.12245.pdf
            modified to be only RY - CNOT - RY

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
        for qubit_index in range(self.number_of_qubits):

            qubit_parameters = parameters[qubit_index : (qubit_index + 1)]

            circuit += RY(qubit_parameters[0])(qubit_index)

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
                circuit_layer += CNOT(i, i+1)
        
       # qubit_ids = list(range(self.number_of_qubits))
       # # Add CNOT(x, x+1) for x in all(qubits)
       # for control, target in zip(
       #     qubit_ids[:-2:], qubit_ids[1::]
       # ):  # loop over qubits 0, 1, 2, 3,...
       #     circuit_layer += CNOT(control, target)

        # Add RY(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer,
            parameters[ self.number_of_qubits : 2 * self.number_of_qubits],
        )

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
        return self.number_of_qubits * 2

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
