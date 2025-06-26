#!/usr/bin/env python3

from scf_guess_datasets import QM9Isomeres, solve
import numpy as np

ds = dataset = QM9Isomeres("cache", size=2)
# ds.build()

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
print(status)  # Status(converged=True, iterations=18)
