################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from typing import Optional, Union

import numpy as np
import sympy
from zquantum.core.circuits import Circuit, X
from zquantum.core.evolution import time_evolution
from zquantum.core.openfermion.ops import (
    FermionOperator,
    InteractionOperator)
from zquantum.core.openfermion.transforms import(
    get_fermion_operator,
    jordan_wigner,
    bravyi_kitaev,
    symmetry_conserving_bravyi_kitaev
)


def exponentiate_fermion_operator(
    fermion_generator: Union[FermionOperator, InteractionOperator],
    transformation: str = "Jordan-Wigner",
    number_of_qubits: Optional[int] = None,
) -> Circuit:
    """Create a circuit corresponding to the exponentiation of an operator.
        Works only for antihermitian fermionic operators.

    Args:
        fermion_generator: fermionic generator.
        transformation: The name of the qubit-to-fermion transformation to use.
        number_of_qubits: This can be used to force the number of qubits in
            the resulting operator above the number that appears in the input operator.
            Defaults to None and the number of qubits in the resulting operator will
            match the number that appears in the input operator.
    """
    if transformation not in ["Jordan-Wigner", "Bravyi-Kitaev"]:
        raise RuntimeError(f"Unrecognized transformation {transformation}")

    # Transform generator to qubits
    if transformation == "Jordan-Wigner":
        qubit_generator = jordan_wigner(fermion_generator)
    else:
        if isinstance(fermion_generator, InteractionOperator):
            fermion_generator = get_fermion_operator(fermion_generator)
        qubit_generator = bravyi_kitaev(fermion_generator, n_qubits=number_of_qubits)

    for term in qubit_generator.terms:
        if isinstance(qubit_generator.terms[term], sympy.Expr):
            if sympy.re(qubit_generator.terms[term]) != 0:
                raise RuntimeError(
                    "Transformed fermion_generator is not anti-hermitian."
                )
            qubit_generator.terms[term] = sympy.im(qubit_generator.terms[term])
        else:
            if not np.isclose(qubit_generator.terms[term].real, 0.0):
                raise RuntimeError(
                    "Transformed fermion_generator is not anti-hermitian."
                )
            qubit_generator.terms[term] = float(qubit_generator.terms[term].imag)
    qubit_generator.compress()

    # Quantum circuit implementing the excitation operators
    circuit = time_evolution(qubit_generator, 1, method="Trotter", trotter_order=1)

    return circuit


def build_hartree_fock_circuit(
    number_of_qubits: int,
    number_of_alpha_electrons: int,
    number_of_beta_electrons: int,
    transformation: str,
    spin_ordering: str = "interleaved",
) -> Circuit:
    """Creates a circuit that prepares the Hartree-Fock state.

    Args:
        number_of_qubits: the number of qubits in the system.
        number_of_alpha_electrons: the number of alpha electrons in the system.
        number_of_beta_electrons: the number of beta electrons in the system.
        transformation: the Hamiltonian transformation to use.
        spin_ordering: the spin ordering convention to use. Defaults to "interleaved".

    Returns:
        zquantum.core.circuit.Circuit: a circuit that prepares the Hartree-Fock state.
    """
    if spin_ordering != "interleaved":
        raise RuntimeError(
            f"{spin_ordering} is not supported at this time. Interleaved is the only"
            "supported spin-ordering."
        )
    circuit = Circuit(n_qubits=number_of_qubits)

    alpha_indexes = list(range(0, number_of_qubits, 2))
    beta_indexes = list(range(1, number_of_qubits, 2))
    index_list = []
    for index in alpha_indexes[:number_of_alpha_electrons]:
        index_list.append(index)
    for index in beta_indexes[:number_of_beta_electrons]:
        index_list.append(index)
    
    if transformation == "qiskit":
        #ordering needed to use qiskit occupation scheme 
        #A-A-_-_-B-B_-_
        
        #clear the old list
        index_list = []

        #REMEMBER THAT THE ORDER IS REVERSED!
        #alpha spins section 
        ialpha=0
        for i in range(number_of_qubits,number_of_qubits//2,-1):
            if ialpha<number_of_alpha_electrons:
                index_list.append(i-1)
                ialpha+=1
                
        #beta spins section
        ibeta=0
        for i in range(number_of_qubits//2,0,-1):
            if ibeta<number_of_beta_electrons:
                index_list.append(i-1)   
                ibeta+=1
        ###index_list=[1,3]
   
    index_list.sort()       
    op_list = [(x, 1) for x in index_list]
    fermion_op = FermionOperator(tuple(op_list), 1.0)
    if transformation == "Jordan-Wigner" or transformation == "qiskit" :
        transformed_op = jordan_wigner(fermion_op)
        qubit_set=np.zeros(number_of_qubits) # Array to keep track of occupied qubits
    elif transformation == "Bravyi-Kitaev":
        transformed_op = bravyi_kitaev(fermion_op, n_qubits=number_of_qubits)
        qubit_set=np.zeros(number_of_qubits) # Array to keep track of occupied qubits
    elif transformation == "BK-2qbr":
        #Setting number of orbitals (qubits)
        active_orbitals=number_of_qubits
        #Setting number of electrons (fermions)
        active_fermions=number_of_alpha_electrons+number_of_beta_electrons
        transformed_op = symmetry_conserving_bravyi_kitaev(fermion_op,
                                                           active_orbitals=active_orbitals,
                                                           active_fermions=active_fermions)
        #BK reduction removes two qubits!
        circuit = Circuit(n_qubits=(number_of_qubits-2))
        qubit_set=np.zeros(number_of_qubits-2) # Array to keep track of occupied qubits

    else:
        raise RuntimeError(
            f"{transformation} is not a supported transformation. Jordan-Wigner, "
            "Bravyi-Kitaev and reduced Bravyi-Kitaev are supported at this time."
        )
    term = next(iter(transformed_op.terms.items()))
    print(term)
    
    for op in term[0]:
        #
        # This is a departure from previous code that woudl assign an X gate for all "non Z" gates.
        # The approach is still in place for JW and BK.
        # However that heuristic doesn't work for reduced BK transformations, so we have to be more 
        # careful and ensure that we are not doubling the number of X gates
        #
        if (op[1] != "Z"):
            # Check if the X operator is already set for this qubit (avoids double setting XX)
            if(qubit_set[op[0]]==0):
                circuit += X(op[0])
                qubit_set[op[0]]+=1
    
    # Print list of "switched on" qubits
    print(qubit_set)
       
    return circuit
