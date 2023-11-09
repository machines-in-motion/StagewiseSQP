# Stagewise implementation of SQPs for OCP

This is the repository that contains the code to generate the benchmarks of [paper]. It includes trajectory optimization scripts and the MPC scipts used in our hardware experiments. 

## Quick Overview of the code

The two main solver variants used in this repository are available in our optimal control solver library [mim_solvers](https://github.com/machines-in-motion/mim_solvers/tree/main) :
1. **Stagewise SQP (SSQP)** : solver that is designed to solve unconstrained optimal control problems quickly. The implementation of the algorithm is in `python/sqp_ocp/solvers/ssqp.py` .
2. **Constrained Stagewise SQP (CSSQP)** : solver that can handle constraints for OCPs by exploiting sparsity.  The implementation of the algorithm is in `python/sqp_ocp/solvers/cssqp.py` .

## Dependencies
- [mim_solvers](https://github.com/machines-in-motion/mim_solvers/tree/main) : C++/Python implementations of the aforementioned solvers
- [crocoddyl](https://github.com/loco-3d/crocoddyl/tree/master) (>= 2.0) : library of tools for optimal control  
- [croco_mpc_utils](https://github.com/machines-in-motion/croco_mpc_utils.git) : helpers for easy & modular prototyping using Crocoddyl
- [pinocchio](https://github.com/stack-of-tasks/pinocchio) : rigid-body dynamics computations
- [mim_solvers](https://github.com/machines-in-motion/mim_robots)
- [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py)

TODO: separate dependencies for hardware experiments and for reproducing the benchmarks

## Maintainers 

The code is maintained by Avadesh Meduri, Armand Jordana, Sébastien Kleff. 

## Paper Citation

Please cite this paper as reference for the code and algorithm. 
