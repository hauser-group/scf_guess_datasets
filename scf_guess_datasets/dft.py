from __future__ import annotations

from .sample import Sample, Status
from pyscf.gto import Mole
from pyscf.gto.basis import load
from pyscf.dft import RKS, UKS
from typing import Any
from numpy.typing import NDArray


def build_basis(name_or_path: str, elements: list[str]) -> dict:
    return {e: load(name_or_path, e) for e in elements}


def build_solver(mole: Mole, functional: str) -> Any:
    method = RKS if mole.multiplicity == 1 else UKS
    solver = method(mole)
    solver.xc = functional
    return solver


def solve(
    solver: Any, guess: NDArray | None
) -> tuple[NDArray, NDArray, NDArray, NDArray, Status]:
    if guess is None:
        solver.run(init_guess="minao")
    else:
        solver.kernel(dm0=guess)

    return (
        solver.get_ovlp(),
        solver.get_hcore(),
        solver.make_rdm1(),
        solver.get_fock(),
        Status(solver.converged, solver.cycles),
    )


def build_solution(path: str, solver: Any, fail: bool):
    overlap, hcore, density, fock, status = solve(solver, None)

    if fail and not status.converged:
        raise RuntimeError("DFT calculation did not converge")

    Sample.save(path, overlap, hcore, density, fock, status)


def build_guess(path: str, solver: Any, scheme: str):
    overlap = solver.get_ovlp()
    hcore = solver.get_hcore()
    density = solver.get_init_guess(key=scheme)
    fock = solver.get_fock(dm=density)

    _, _, _, _, status = solve(solver, density)
    Sample.save(path, overlap, hcore, density, fock, status)
