# SCF Guess Datasets

This package provides a number of molecular datasets to be used in a ML context.
Each dataset represents a set of randomly selected molecules from an original
collection of xyz files, augmented with tensor quantities obtained via DFT
calculations.

## Datasets

For a description of each dataset see [quantum-machine.org](http://quantum-machine.org/datasets/).

- [Qm9](http://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) 134k small organic molecules of CHONF. Molecules are non-charged and closed-shell. Calculated at B3LYP/6-31G(2df,p) level.
- [Qm9Isomeres](http://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) 6k constitutional isomeres from C7H10O2 taken from Qm9. Calculated at B3LYP/6-31G(2df,p) level.
- [Qm9IsomeresMd](http://quantum-machine.org/data/c7o2h10_md.tar.gz) Molecular trajectories of 113 molecules at 500 K randomly selected from Qm9 Isomeres. Calculated at the B3LYP/6-31G(2df,p) level.

## Installation

- Clone this repository to `scf_guess_datasets`
- Invoke `cd scf_guess_datasets && pip install -e .`

## How to use

```python
from scf_guess_datasets import Qm9Isomeres

dataset = Qm9Isomeres(
    "/home/bob/datasets", # data stored in /home/bob/datasets/qm9_isomeres
    size=10, # number of molecules (optional, just for testing)
    val=0.1, # fraction of validation samples (optional, just for testing)
    test=0.1 # fraction of test samples (optional, just for testing)
)

dataset.build() # just once, omit if /home/bob/datasets/qm9_isomeres exists

for key in dataset.train_keys:  # same for val_keys
    sample = dataset.solution(key)  # dft result for that molecule

    print(sample.overlap)  # NDArray from PysCF
    print(sample.hcore)  # NDArray from PysCF
    print(sample.density)  # NDArray from PysCF
    print(sample.fock)  # NDArray from PysCF

    print(sample.status)  # Status(converged=True, iterations=11)

    for scheme, sample in dataset.guesses(key).items():
        # sample has same structure as returned by dataset.solution
        # matrices correspond to the initial guess
        # status describes calculation starting from guess

        print(scheme, sample.status)

# Let's score some custom-made density matrix for a given molecule

from scf_guess_datasets import solve
import numpy as np

solver = dataset.solver(3)  # obtain a new solver for molecule 3
guess = np.ones_like(solver.get_ovlp())
overlap, hcore, density, fock, status = solve(solver, guess)
print(density)  # the converged density
print(status)  # Status(converged=True, iterations=19)
```

## Structure

Each dataset provided by this package implements the `scf_guess_datasets.Dataset`
interface. A single implementation is represented by an individual package,
containing a `xyz` directory  as well as an optional `basis.gbs` basis set file.
In order to add a new dataset, create a new subpackage for it and adapt to your
needs, e.g. by specifying a custom basis or functional.