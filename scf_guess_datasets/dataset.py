from __future__ import annotations

from .sample import Sample
from .dft import build_basis, build_solver, build_solution, build_guess
from abc import ABC, abstractmethod
from importlib.resources import files
from functools import cached_property
from typing import Any
from pyscf.gto import Mole
from warnings import warn
from pathlib import Path
from math import floor

import random
import shutil
import pickle


class Dataset(ABC):
    def __init__(
        self,
        data_directory: str,
        name: str,
        basis: str | None,
        size: int,
        split_ratio: float,
    ) -> None:
        self.name, self.size, self.split_ratio = name, size, split_ratio

        base = files(f"scf_guess_datasets.{name}")
        self.xyz = f"{base}/xyz"
        self.basis = basis or f"{base}/basis.gbs"

        self.data = f"{data_directory}/{name}"

    @cached_property
    def names(self) -> list[str]:
        names = sorted(xyz.stem for xyz in Path(self.xyz).glob("*.xyz"))

        random.seed(0)
        random.shuffle(names)

        return names

    @cached_property
    def keys(self) -> list[int]:
        with open(f"{self.data}/keys.pkl", "rb") as f:
            return pickle.load(f)

    @cached_property
    @abstractmethod
    def schemes(self) -> list[str]:
        pass

    @cached_property
    def train_keys(self) -> list[int]:
        return self.keys[: floor(self.size * self.split_ratio)]

    @cached_property
    def val_keys(self) -> list[int]:
        train = len(self.train_keys)
        val = self.size - train
        return self.keys[train : train + val]

    @cached_property
    @abstractmethod
    def elements(self) -> list[str]:
        pass

    @cached_property
    def basis_set(self) -> Any:
        return build_basis(self.basis, self.elements)

    @cached_property
    @abstractmethod
    def functional(self) -> str:
        pass

    @abstractmethod
    def molecule(self, key: int) -> Mole:
        pass

    def solver(self, key: int) -> Any:
        return build_solver(self.molecule(key), self.functional)

    def solution(self, key: int) -> Sample:
        return Sample(f"{self.data}/{key}")

    def guesses(self, key: int) -> dict[str, Sample]:
        return {s: Sample(f"{self.data}/{key}/{s}") for s in self.schemes}

    def build(self):
        data = Path(self.data)

        assert not data.exists(), f"refusing to build {self.name} twice"
        data.mkdir(parents=True, exist_ok=True)

        keys = []

        for key, name in enumerate(self.names):
            print(f"Building sample {key} / {self.size - 1} ({name})")

            try:
                solver = self.solver(key)
                build_solution(f"{self.data}/{key}", solver, fail=True)

                for scheme in self.schemes:
                    print(f"Building guess for scheme {scheme}")
                    solver = self.solver(key)
                    build_guess(f"{self.data}/{key}/{scheme}", solver, scheme)

                keys.append(key)
            except Exception as e:
                warn(f"Unable to build sample from {name}: {e}")
                shutil.rmtree(Path(f"{self.data}/{key}"))

            if len(keys) >= self.size:
                break

        with open(f"{self.data}/keys.pkl", "wb") as f:
            pickle.dump(keys, f)

        assert len(keys) == self.size
