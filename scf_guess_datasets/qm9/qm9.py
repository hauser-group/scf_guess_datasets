from ..dataset import Dataset
from ..dft import build_solution, build_guess
from functools import cached_property
from pathlib import Path
from pyscf.gto import Mole, M
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from random import shuffle
from warnings import warn

import numpy as np
import pickle
import shutil

class Qm9(Dataset):
    def __init__(
        self,
        data_directory: str,
        size: int = 500,
        val: float = 0.1,
        test: float = 0.1,
    ) -> None:
        super().__init__(data_directory, "qm9", None, size, val, test)

    @cached_property
    def schemes(self) -> list[str]:
        # huckel and mod_huckel won't work since we don't have ECPs
        return ["1e", "vsap", "sap", "minao"]

    @cached_property
    def elements(self) -> list[str]:
        return ["H", "C", "O", "N", "F"]

    def molecule(self, key: int) -> Mole:
        name = self.names[key]
        xyz = str(Path(self.xyz).joinpath(f"{name}.xyz"))

        return M(  # QM9 contains non-charged, closed-shell molecules only
            atom=xyz,
            basis=self.basis_set,
            charge=0,
            spin=0,
            verbose=0,
            symmetry=False,
            cart=False,
        )

    @cached_property
    def functional(self) -> str:
        return "B3LYPG"

    def build(self):
        data = Path(self.data)

        assert not data.exists(), f"refusing to build {self.name} twice"
        data.mkdir(parents=True, exist_ok=True)

        keys, sizes = [], []

        for key, name in enumerate(self.names):
            try:
                sizes.append(self.molecule(key).natm)
                keys.append(key)
            except Exception as e:
                warn(f"Unable to build sample from {name}: {e}")

            print(key)

        keys, sizes = np.array(keys), np.array(sizes)

        bins = np.linspace(min(sizes), max(sizes), 14)
        bin_labels = np.digitize(sizes, bins, right=True)

        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=(self.val + self.test) / self.size, random_state=0
        )

        train_idx, val_test_idx = next(stratifier.split(keys, bin_labels))

        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test / (self.val + self.test), random_state=0
        )

        val_idx, test_idx = next(
            stratifier.split(keys[val_test_idx], bin_labels[val_test_idx])
        )

        train = self._build_set(self.train, keys[train_idx], bin_labels[train_idx])
        val = self._build_set(self.val, keys[val_idx], bin_labels[val_idx])
        test = self._build_set(self.test, keys[test_idx], bin_labels[test_idx])

        keys = train + val + test

        with open(f"{self.data}/keys.pkl", "wb") as f:
            pickle.dump(keys, f)

        assert len(keys) == self.size

    def _build_set(self, size: int, keys: list[int], bin_labels: list[int]) -> list[int]:
        # We will first try to uniformly sample from all bins to achieve the total size
        # If some bins contain too few samples, we compensate by randomly sampling from
        # other bins

        bins = defaultdict(list)

        for i in range(len(keys)):
            bins[bin_labels[i]].append(keys[i])

        n_bins = len(bins.keys())
        n_bin = size // n_bins

        uniform, remainder = [], []
        for i, keys in bins.items():
            uniform.extend(keys[:n_bin])
            remainder.extend(keys[n_bin:])

        shuffle(remainder)
        candidates = uniform + remainder

        result = []

        for i, key in enumerate(candidates):
            name = self.names[key]
            print(f"Building sample {i} / {size - 1} ({name})")

            try:
                solver = self.solver(key)
                build_solution(f"{self.data}/{key}", solver, fail=True)

                for scheme in self.schemes:
                    print(f"Building guess for scheme {scheme}")
                    solver = self.solver(key)
                    build_guess(f"{self.data}/{key}/{scheme}", solver, scheme)

                result.append(key)
            except Exception as e:
                warn(f"Unable to build sample from {name}: {e}")
                shutil.rmtree(Path(f"{self.data}/{key}"))

            if len(result) >= size:
                break

        return result
