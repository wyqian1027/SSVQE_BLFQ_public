# SSVQE-BLFQ public repository

WQ 06/26/2022

## About

The Subspace-search variational quantum eigensolver (SSVQE) algorithm is a recently proprosed VQE variant that enables finding the full hadronic spectroscopy in nuclear physics. The algorithm is described in [Subspace-search variational quantum eigensolver for excited states](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033062)

This repository constains source code and demos on using SSVQE to solve hadronic observables in BLFQ. See a recent application in [Solving hadron structures using the basis light-front quantization approach on quantum computers](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043193)

## Demos

* SSVQE optimizations: demo_ssvqe_basic.ipynb
* SSVQE BLFQ observables: demo_ssvqe_blfq_observables.ipynb

### Installation tips

1. For Qiskit installation, follow the [official guide](https://qiskit.org/documentation/getting_started.html). Be sure to install both qiskit and qiskit[visualization]. Usually, it is recommended to install Python libraries in a virtual environment.

2. To use Qiskit for physical problems, install qiskit[nature]; see [official doc](https://qiskit.org/ecosystem/nature/index.html)

3. To use Qiskit for optimization such as VQE, install qiskit[optimization]; see [official doc](https://qiskit.org/ecosystem/optimization/)

4. Related tutorials on Qiskit Nature and Optimization: [Qiskit nature tutorials](https://qiskit.org/ecosystem/nature/tutorials/index.html), [Qiskit optimization tutorials](https://qiskit.org/ecosystem/optimization/tutorials/index.html) 
 
5. Other competiting algorithms, such as the VQD approach (arXiv:2002.11724), already implemented in Qiskit. See [tutorial](https://qiskit.org/documentation/tutorials/algorithms/04_vqd.html)

