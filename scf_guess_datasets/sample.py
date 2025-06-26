from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from numpy.typing import NDArray
from numpy import load, save
from pathlib import Path

import pickle


@dataclass(frozen=True)
class Status:
    converged: bool
    iterations: int | None


class Sample:
    def __init__(self, path: str):
        self.path = path

    @cached_property
    def overlap(self) -> NDArray:
        return load(f"{self.path}/overlap.npy")

    @cached_property
    def hcore(self) -> NDArray:
        return load(f"{self.path}/hcore.npy")

    @cached_property
    def density(self) -> NDArray:
        return load(f"{self.path}/density.npy")

    @cached_property
    def fock(self) -> NDArray:
        return load(f"{self.path}/fock.npy")

    @cached_property
    def status(self) -> Status:
        with open(f"{self.path}/status.pkl", "rb") as f:
            return pickle.load(f)

    @classmethod
    def save(
        cls,
        path: str,
        overlap: NDArray,
        hcore: NDArray,
        density: NDArray,
        fock: NDArray,
        status: Status,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

        save(f"{path}/overlap.npy", overlap)
        save(f"{path}/hcore.npy", hcore)
        save(f"{path}/density.npy", density)
        save(f"{path}/fock.npy", fock)

        with open(f"{path}/status.pkl", "wb") as f:
            pickle.dump(status, f)
