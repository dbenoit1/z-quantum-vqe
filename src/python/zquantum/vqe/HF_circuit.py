################################################################################
# © Copyright 2021 Zapata Computing Inc.
# DMB mods to [theta] - [-theta] - [CNOT] 04/05/2022
# TESTED WITH STATEVECTOR SIMS 06/05/2022 - works
# Now supports JW/BK/BK-2qbr
################################################################################
from typing import List, Optional 

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import X,CNOT, RY, Circuit
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import ansatz_property

from .utils import build_hartree_fock_circuit

class HF_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")
    transformation = ansatz_property("transformation")


    def __init__(self, number_of_layers: int, 
                 number_of_qubits: int, 
                 nb_occ: int, 
                 transformation: str = "Jordan-Wigner"
                ):
        """An ansatz implementation for the Hardware Efficient Quantum Compiling Ansatz
            used in https://arxiv.org/pdf/2011.12245.pdf
            modified to be only Ry(+theta) => Ry(-Theta) => IdCNOT

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
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be a positive integer")
        super().__init__(number_of_layers)
       # assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ
        self._transformation = transformation

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

    def _build_circuit_layer(self, parameters: np.ndarray,occupied_qubit_list) -> Circuit:
        """Build circuit layer for the hardware efficient HF ansatz analog
 
        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Quantum
            Compiling Ansatz
        """
        circuit_layer = Circuit()
        
        # Add RY(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer, parameters[: self.number_of_qubits]
        )

        # Add CNOT(x, x+1) for x in all(qubits)
        #Slightly more hardwired approach:
        for i in range(self.number_of_qubits):
            target=i+1
            if (target<self.number_of_qubits):
                 if(occupied_qubit_list[i]==1):
                    #each occupied state is a X so use X-CNOT-X
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
        
        #Start with HF state
        original_nb_qubits=self.number_of_qubits
        #check for reduced BK transformation (needs an extra 2 qubits first)
        if (self.transformation=='BK-2qbr'):
            original_nb_qubits+=2
            
        circuit += build_hartree_fock_circuit(number_of_qubits=original_nb_qubits,
            number_of_alpha_electrons=self.nb_occ//2,number_of_beta_electrons=self.nb_occ//2,
            transformation=self.transformation,
        )
        
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

       # # Hardwired JW HF ansatz instead (previous one has library issues)
       # for i in range(self.nb_occ):
       #     #adds a not (i.e. |1>) for each occupied state starting from 0 up to nb_occ (-1 coz python.. )
       #     circuit += X(i)

        #Then layers of R[+theta]-R[-theta]-CNOTS-
        for layer_index in range(self.number_of_layers):
            circuit += self._build_circuit_layer(
                parameters[
                    layer_index
                    * self.number_of_params_per_layer : (layer_index + 1)
                    * self.number_of_params_per_layer
                ],
                occupied_qubit_list
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
