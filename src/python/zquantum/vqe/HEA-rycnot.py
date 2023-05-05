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

from .utils import build_hartree_fock_circuit

class HEA_RY_CNOT_RY_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")
    transformation = ansatz_property("transformation")
    list_occupied_qubit = ansatz_property("list_occupied_qubit")

    def __init__(self, number_of_layers: int,
                 number_of_qubits: int, 
                 nb_occ: int, 
                 transformation: str = "Jordan-Wigner",
                ):
        """An ansatz implementation of the Hardware Efficient Ansatz
            used in 10.1021/acs.jctc.1c00091
            -HF - RY - [CNOT - RY]n -

        Args:
            number_of_layers: number of layers in the circuit.
            number_of_qubits: number of qubits in the circuit.
            nb_occ: number of occupied states (spin orbitals, for example)
            tranformation: Mapping transformation (JW/BK/BK-2qbr), default JW

        Attributes:
            number_of_qubits: See Args
            number_of_layers: See Args
            nb_occ: See Args
            transformation: string
        """
        if number_of_layers < 0:
            raise ValueError("number_of_layers must be a positive integer")
        super().__init__(number_of_layers)
        #assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ
        self._transformation = transformation
        print(number_of_qubits, nb_occ)

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
        """Build circuit layer for the hardware efficient ansatz

        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Ansatz
        """
        circuit_layer = Circuit()
        
        # Add CNOT(x, x+1) for x in all(qubits)
        #Slightly more hardwired approach:
        for i in range(self.number_of_qubits):
            target=i+1
            if (target<self.number_of_qubits):
                circuit_layer += CNOT(i, i+1)
     
        # Add RY(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer,
            parameters[ 0 : self.number_of_qubits],
        )

        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: 10.1021/acs.jctc.1c00091, Fig. 1)

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = np.asarray(self.symbols, dtype=object)

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
                
        ## Hardwired JW HF ansatz instead (previous one has library issues)
        ## set to zero to bypass HF init
        #for i in range(self.nb_occ):
        #    #adds a not (i.e. |1>) for each occupied state starting from 0 up to nb_occ (-1 coz python.. )
        #    circuit += X(i)
        
        #Start with HF state is number of electrons are given
        if (self.nb_occ>0):
            original_nb_qubits=self.number_of_qubits
            #check for reduced BK transformation (needs an extra 2 qubits first)
            if (self.transformation=='BK-2qbr'):
                original_nb_qubits+=2
            
            circuit += build_hartree_fock_circuit(number_of_qubits=original_nb_qubits,
                number_of_alpha_electrons=self.nb_occ//2,number_of_beta_electrons=self.nb_occ//2,
                transformation=self.transformation)
            print("RY-CNOT-RY ansatz HF start")
            print(circuit)
             #Keep track of which qubits are occupied (X) in occupied_qubit_list (0 is unocupied, 1 occupied)
            occupied_qubit_list=np.zeros(self.number_of_qubits)
        
            print(circuit)
        
            #This loops over the operations in the circuit and those whould only be Xgates for the occupied qubits
            for gates in circuit.operations:
             # extract occupied qubit index
                myval=gates.qubit_indices[0]
                # set occupied marker to 1 to represent its state
                if (occupied_qubit_list[myval]==0):
                # Here we are just being careful that we dont erase a qubit that was already set
                # was 0 and now flipped to 1
                    occupied_qubit_list[myval]=1
                    print("X operation on qubit: "+str(myval))
           
            print("full occupation map")
            print(occupied_qubit_list)

        # Add RY(theta)
        circuit = self._build_rotational_subcircuit(
            circuit, parameters[0: self.number_of_qubits]
        )
        
        if (self.number_of_layers>0):
            for layer_index in range(self.number_of_layers):
                circuit += self._build_circuit_layer(
                parameters[self.number_of_qubits+
                    layer_index
                    * self.number_of_params_per_layer : self.number_of_qubits+(layer_index + 1)
                    * self.number_of_params_per_layer
                    ]
                )
                
        #print circuit?
        print(circuit)
        
        return circuit

    @property
    def number_of_params(self) -> int:
        """
        Returns number of parameters in the ansatz.
        """
        return self.number_of_qubits+self.number_of_params_per_layer * self.number_of_layers

    @property
    def number_of_params_per_layer(self) -> int:
        """
        Returns number of parameters in the layer.
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
