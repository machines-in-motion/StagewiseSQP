# Stagewise implementation of SQPs for OCP

This is the repository that contains the python implementations of the sparse SQP designed for optimal control. The code base was used to generate the results discussed in the paper referenced before. 

## Quick Overview of the code

The two main solver variants in this repository are the
1. **Stagewise SQP (SSQP)** : solver that is designed to solve unconstrained optimal control problems quickly. The implementation of the algorithm is in `python/sqp_ocp/solvers/ssqp.py` .
2. **Constrained Stagewise SQP (CSSQP)** : solver that can handle constraints for OCPs by exploiting sparsity.  The implementation of the algorithm is in `python/sqp_ocp/solvers/cssqp.py` .

Both these solvers are implementated by using Crocoddyl as the base software. Consquently, one can continue constructing problems as before using Crocoddyl and can choose to use these solvers to solve the problem. Examples of how to use the solvers are in the examples directory.

## Code Update  
You need to compile [our branch of Crocoddyl 1.9](https://github.com/machines-in-motion/crocoddyl/tree/gnms) for now in order to reproduce our benchmarks and experiments. We will release soon a C++ implementation of these solvers compatible with Crocoddyl 2.0 in [mim_solvers](https://github.com/machines-in-motion/mim_solvers/tree/main). Please keep track of this repo for updates ! 

## Maintainers 

The code is maintained by Avadesh Meduri, Armand Jordana, Sébastien Kleff. 

## Paper Citation

Please cite this paper as reference for the code and algorithm. 
