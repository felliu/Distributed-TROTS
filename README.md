# Parallel Optimization for TROTS
## About
This repository contains code to run (possibly distributed) optimization for radiation therapy problems from the [TROTS dataset](https://sebastiaanbreedveld.nl/trots/). It provides a C++-library which computes (slightly modified) objective functions and gradients from the TROTS dataset, which can be used to interface to general optimization solvers.

Furthermore, there are drivers to run the optimization using the [IPOPT](https://github.com/coin-or/Ipopt) optimization solver, including one where the evaluation of the objective function is parallelized using MPI, making it capable of utilizing distributed computational resources

## Usage
### Dependencies

Dependencies vary depending on which parts of the code are required. In general, a C++-17 capable compiler is required, as well as CMake to build the code.

**trots_lib**  
Library for computing objective functions, constraints and their gradients.

* Intel MKL **or** (tentative support) Eigen
* matio (provided as a git submodule)
    * matio requires HDF5 to work with Matlab v7.3 files

In our experience, recent versions of Intel MKL perform the best, both on Intel and AMD CPUs. Since **trots_lib** is used by all other parts of the codes, this dependency is required by every other part too.

**ipopt_driver**  
Serial driver to optimize TROTS problems using the IPOPT optimization solver.

*  [IPOPT](https://github.com/coin-or/Ipopt)

**ipopt_mpi_driver**  
Parallel driver (MPI) to optimize TROTS problems using the IPOPT optimization solver

* [IPOPT](https://github.com/coin-or/Ipopt)
* MPI
* Boost Serialization module

### Building

Building is done in a standard CMake fashion. Flags specific to this package:
```
-DUSE_MKL=true #Selects Intel MKL as the sparse matrix library
-DMPI=true #Build the target with MPI parallelization
```
The code can be built simply by navigating to the root directory and executing something like
```
mkdir build
cd build
cmake <flags> ..
cmake --build .
```


