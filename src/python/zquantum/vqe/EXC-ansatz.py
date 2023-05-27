################################################################################
# © Copyright 2021 Zapata Computing Inc.
# DMB implementation of the excitation presertving ansatz
# 09/05/22 DMB: NOT TESTED WITH STATEVECTOR 
################################################################################
from typing import List, Optional

import numpy as np
import sympy
from overrides import overrides
from zquantum.core.circuits import XX,YY, RZ ,RY, CNOT, X, Circuit
from zquantum.core.interfaces.ansatz import Ansatz
from zquantum.core.interfaces.ansatz_utils import ansatz_property

from .utils import build_hartree_fock_circuit

class EXC_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")
    transformation = ansatz_property("transformation")

    def __init__(self, number_of_layers: int,
                 number_of_qubits: int, 
                 nb_occ: int, 
                 transformation: str = "Jordan-Wigner",
                ):
        """An ansatz implementation of the excitation preserving Ansatz
            used in 
            -HF - [RXX+RYY (XY)- Rz]n 

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
        if transformation not in ["Jordan-Wigner"]:
            raise RuntimeError(f"ONLY WORKS FOR JORDAN-WIGNER TRANSFORMATION, BUT YOU REQUESTED {transformation}")
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

            circuit += RZ(qubit_parameters[0])(qubit_index)

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
            target=i+2
            if (target<self.number_of_qubits):
                qubit_parameters = parameters[i : (i + 1)]
                circuit_layer += XX(qubit_parameters[0])(i, target)
                circuit_layer += YY(qubit_parameters[0])(i, target)
     
        # Add RZ(theta)
        circuit_layer = self._build_rotational_subcircuit(
            circuit_layer,
            parameters[ self.number_of_qubits-2 : (2*self.number_of_qubits)-2],
        )

        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: )

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = np.asarray(self.symbols, dtype=object)

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
                
        #Start with HF state is number of electrons are given
        if (self.nb_occ>0):
            original_nb_qubits=self.number_of_qubits
            #check for reduced BK transformation (needs an extra 2 qubits first)
            if (self.transformation=='BK-2qbr'):
                original_nb_qubits+=2
            
            circuit += build_hartree_fock_circuit(number_of_qubits=original_nb_qubits,
                number_of_alpha_electrons=self.nb_occ//2,number_of_beta_electrons=self.nb_occ//2,
                transformation=self.transformation)
            print("RZ-RXX+RYY-RZ ansatz HF start")
            print(circuit)
        
        #REMOVED INITIAL RZ AS A FINAL RZ INSTEAD WORKS JUST AS WELL ON SIMS
        # Add RZ(theta)
        #circuit = self._build_rotational_subcircuit(
        #    circuit, parameters[0: self.number_of_qubits]
        #)
        
        if (self.number_of_layers>0):
            for layer_index in range(self.number_of_layers):
                circuit += self._build_circuit_layer(
                parameters[layer_index * self.number_of_params_per_layer : 
                    (layer_index + 1) * self.number_of_params_per_layer ]    
               # parameters[self.number_of_qubits+
               #     layer_index
               #     * self.number_of_params_per_layer : self.number_of_qubits+(layer_index + 1)
               #     * self.number_of_params_per_layer
               #     ]
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
        return self.number_of_qubits-2+self.number_of_qubits 
    
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

    
### ASWAP ANSATZ CLASS
class ASWAP_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")
    transformation = ansatz_property("transformation")

    def __init__(self, number_of_layers: int,
                 number_of_qubits: int, 
                 nb_occ: int, 
                 transformation: str = "Jordan-Wigner",
                ):
        """An ansatz implementation of the ASWAP excitation preserving Ansatz
            used in Gard+ 2020 (https://doi.org/10.1038/s41534-019-0240-1)
            
            -HF - [Cnot - Ry Rz - Cnot - RZ Ry - Cnot]n (A gate) 

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
        if transformation not in ["Jordan-Wigner"]:
            raise RuntimeError(f"ONLY WORKS FOR JORDAN-WIGNER TRANSFORMATION, BUT YOU REQUESTED {transformation}")
        assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ
        self._transformation = transformation
        print(number_of_qubits, nb_occ)

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        """Build circuit layer for the ASWAP hardware efficient ansatz

        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Ansatz
        """
        circuit_layer = Circuit()
        
        for i in range(self.number_of_qubits):
            target=i+2
            if (target<self.number_of_qubits):
                qubit_parameters = parameters[2*i : (2*i + 2)]
                #Ry rotation parameter
                t=qubit_parameters[0]+np.pi/2.
                #Rz rotation parameter
                p=qubit_parameters[1]+np.pi

                #inverted CNOT
                circuit_layer += CNOT(target,i)
                
                #R(t,p) dagger = Ry(-t) Rz(-p)
                circuit_layer += RY(-t)(target)
                circuit_layer += RZ(-p)(target)
                
                #standard CNOT
                circuit_layer += CNOT(i,target)
                
                #R(t,p) = Rz(p) Ry(t)
                circuit_layer += RZ(p)(target)
                circuit_layer += RY(t)(target)
                
                #inverted CNOT
                circuit_layer += CNOT(target,i)
  
        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: )

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = np.asarray(self.symbols, dtype=object)

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
                
        #Start with HF state is number of electrons are given
        if (self.nb_occ>0):
            original_nb_qubits=self.number_of_qubits
            circuit += build_hartree_fock_circuit(number_of_qubits=original_nb_qubits,
                number_of_alpha_electrons=self.nb_occ//2,number_of_beta_electrons=self.nb_occ//2,
                transformation=self.transformation)
            print("ASWAP ansatz HF start")
            print(circuit)
        
        if (self.number_of_layers>0):
            for layer_index in range(self.number_of_layers):
                circuit += self._build_circuit_layer(
                parameters[layer_index * self.number_of_params_per_layer : 
                    (layer_index + 1) * self.number_of_params_per_layer ]  )  
                
        print(circuit)
        
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
        Returns number of parameters in the layer.
        """
        return (self.number_of_qubits-2)*2 
    
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
    
### FULL REAL ASWAP ANSATZ CLASS
class RASWAP_Ansatz(Ansatz):

    supports_parametrized_circuits = True
    number_of_qubits = ansatz_property("number_of_qubits")
    nb_occ = ansatz_property("nb_occ")
    transformation = ansatz_property("transformation")

    def __init__(self, number_of_layers: int,
                 number_of_qubits: int, 
                 nb_occ: int, 
                 transformation: str = "Jordan-Wigner",
                ):
        """An ansatz implementation of the ASWAP excitation preserving Ansatz
            used in Gard+ 2020 (https://doi.org/10.1038/s41534-019-0240-1)
            
            -HF - [Cnot - Ry Rz - Cnot - RZ Ry - Cnot]n (A gate) 
            
            USING ONLY REAL GATES (phi=0) AND FULL ENTANGLEMENT

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
        if transformation not in ["Jordan-Wigner"]:
            raise RuntimeError(f"ONLY WORKS FOR JORDAN-WIGNER TRANSFORMATION, BUT YOU REQUESTED {transformation}")
        assert number_of_qubits % 2 == 0
        self._number_of_qubits = number_of_qubits
        self._nb_occ = nb_occ
        self._transformation = transformation
        print(number_of_qubits, nb_occ)

    def _build_circuit_layer(self, parameters: np.ndarray) -> Circuit:
        """Build circuit layer for the ASWAP hardware efficient ansatz

        Args:
            parameters: The variational parameters (or symbolic parameters)

        Returns:
            Circuit containing a single layer of the Hardware Efficient Ansatz
        """
        circuit_layer = Circuit()
        u=0
        for i in range(self.number_of_qubits):
            target=i+2
            if (target<self.number_of_qubits):
                qubit_parameters = parameters[u : (u + 1)]
                #Ry rotation parameter
                t=qubit_parameters[0]+np.pi/2.
#REAL ASWAP GATE LAYER A
                #inverted CNOT
                circuit_layer += CNOT(target,i)
                #R(t,p) dagger = Ry(-t) Rz(-p)
                circuit_layer += RY(-t)(target)
                circuit_layer += RZ(-np.pi)(target)
                #standard CNOT
                circuit_layer += CNOT(i,target)
                #R(t,p) = Rz(p) Ry(t)
                circuit_layer += RZ(np.pi)(target)
                circuit_layer += RY(t)(target)
                #inverted CNOT
                circuit_layer += CNOT(target,i)
                u+=1
                
          for i in range(2,self.number_of_qubits):
            target=i+2
            if (target<self.number_of_qubits):
                qubit_parameters = parameters[u : (u + 1)]
                #Ry rotation parameter
                t=qubit_parameters[0]+np.pi/2.
#REAL ASWAP GATE LAYER B
                #inverted CNOT
                circuit_layer += CNOT(target,i)
                #R(t,p) dagger = Ry(-t) Rz(-p)
                circuit_layer += RY(-t)(target)
                circuit_layer += RZ(-np.pi)(target)
                #standard CNOT
                circuit_layer += CNOT(i,target)
                #R(t,p) = Rz(p) Ry(t)
                circuit_layer += RZ(np.pi)(target)
                circuit_layer += RY(t)(target)
                #inverted CNOT
                circuit_layer += CNOT(target,i)
                u+=1
        return circuit_layer

    @overrides
    def _generate_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """Builds the ansatz circuit (based on: )

        Args:
            params (numpy.ndarray): input parameters of the circuit (1d array).

        Returns:
            Circuit
        """
        if parameters is None:
            parameters = np.asarray(self.symbols, dtype=object)

        assert len(parameters) == self.number_of_params

        circuit = Circuit()
                
        #Start with HF state is number of electrons are given
        if (self.nb_occ>0):
            original_nb_qubits=self.number_of_qubits
            circuit += build_hartree_fock_circuit(number_of_qubits=original_nb_qubits,
                number_of_alpha_electrons=self.nb_occ//2,number_of_beta_electrons=self.nb_occ//2,
                transformation=self.transformation)
            print("ASWAP ansatz HF start")
            print(circuit)
        
        if (self.number_of_layers>0):
            for layer_index in range(self.number_of_layers):
                circuit += self._build_circuit_layer(
                parameters[layer_index * self.number_of_params_per_layer : 
                    (layer_index + 1) * self.number_of_params_per_layer ]  )  
                
        print(circuit)
        
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
        Returns number of parameters in the layer.
        """
        return (self.number_of_qubits-2)+(self.number_of_qubits-4)
    
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
