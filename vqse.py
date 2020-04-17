#!/usr/bin/env python

import multiprocessing, numpy, scipy
from openfermion.hamiltonians import MolecularData
from openfermion.ops import FermionOperator
from openfermion.transforms import get_fermion_operator, get_sparse_operator
from openfermion.utils import (expectation, get_ground_state,
    hermitian_conjugated, normal_ordered)
from openfermionpsi4 import run_psi4

basis = 'cc-pvdz'
multiplicity = 1
data_directory= '.'

active_spatial_orbitals = 2
virtual_spatial_orbitals = 8

active_spin_orbitals = 2*active_spatial_orbitals
virtual_spin_orbitals = 2*virtual_spatial_orbitals

roundoff = 256*numpy.finfo(float).eps

def nonzero_image(operator, state):
    return numpy.linalg.norm(
        get_sparse_operator(
            operator, n_qubits=active_spin_orbitals)*state) > roundoff

def linear_operators(state):
    ra = range(active_spin_orbitals)
    rf = range(active_spin_orbitals + virtual_spin_orbitals)
    aa = [FermionOperator((i, 0)) for i in ra]
    cf = [FermionOperator((i, 1)) for i in rf]
    lo = []
    for i in ra:
        if nonzero_image(aa[i], state):
            for j in rf:
                if (i + j)%2  == 0:
                    lo.append(cf[j]*aa[i])
    return lo

def quadratic_operators(state):
    ra = range(active_spin_orbitals)
    rv = range(virtual_spin_orbitals)
    aa = [FermionOperator((i, 0)) for i in ra]
    cv = [FermionOperator((active_spin_orbitals + i, 1)) for i in rv]
    qo = []
    for i in ra:
        for j in range(i):
            if nonzero_image(aa[i]*aa[j], state):
                for k in rv:
                    for l in range(k):
                        if i%2 + j%2 == k%2 + l%2:
                            qo.append(cv[k]*cv[l]*aa[i]*aa[j])
    return qo

def vqse_operators(state):
    return linear_operators(state) + quadratic_operators(state)

def filter_terms(operator):
    o = FermionOperator()
    for k, v in operator.terms.items():
        if not any(i >= active_spin_orbitals for i, _ in k):
            o += FermionOperator(k, v)
    return o

def expectation_value(operator, state):
    f = filter_terms(normal_ordered(operator))
    return expectation(
        get_sparse_operator(
            f, n_qubits=active_spin_orbitals), state) if len(f.terms) else 0j

def h_matrix(hamiltonian, operators, state):
    l = len(operators)
    h = numpy.empty((l, l), dtype=numpy.complex128)
    for i in range(l):
        for j in range(i + 1):
            o = hermitian_conjugated(operators[i])
            o *= hamiltonian
            o *= operators[j]
            e = expectation_value(o, state)
            h[i, j] = e
            if i != j: h[j, i] = e.conjugate()
    return h

def s_matrix(operators, state):
    l = len(operators)
    s = numpy.empty((l, l), dtype=numpy.complex128)
    for i in range(l):
        for j in range(i + 1):
            o = hermitian_conjugated(operators[i])
            o *= operators[j]
            e = expectation_value(o, state)
            s[i, j] = e
            if i != j: s[j, i] = e.conjugate()
    return s

def generalized_eigenvalues(h, s):
    e, v = numpy.linalg.eigh(s)
    i = e > roundoff
    v = v[:, i]
    s = numpy.diag(e[i].astype(numpy.complex128))
    h = v.conjugate().T @ h @ v
    f = scipy.linalg.eigvalsh(h, s)
    return numpy.sort(f)

def energy_levels(separation):
    g = [('H', (0., 0., 0.)), ('H', (0., 0., separation))]
    d = str(separation)
    m = MolecularData(g, basis, multiplicity, description=d,
                      data_directory=data_directory)
    run_psi4(m, run_fci=True, delete_output=True)
    _, s = get_ground_state(
        get_sparse_operator(
            normal_ordered(
                get_fermion_operator(
                    m.get_molecular_hamiltonian(
                        active_indices=range(active_spatial_orbitals))))))
    h = normal_ordered(
        get_fermion_operator(
            m.get_molecular_hamiltonian(
                active_indices=range(
                    active_spatial_orbitals + virtual_spatial_orbitals))))
    o = vqse_operators(s)
    return generalized_eigenvalues(h_matrix(h, o, s), s_matrix(o, s))

separations = [round(0.4 + 0.05*i, 2) for i in range(53)]

with multiprocessing.Pool() as p:
    results = [p.apply_async(energy_levels, (s,)) for s in separations]
    for s, r in zip(separations, results): print(s, *r.get())
