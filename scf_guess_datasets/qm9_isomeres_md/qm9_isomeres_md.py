from ..dataset import Dataset
from functools import cached_property
from pathlib import Path
from pyscf.gto import Mole, M


class Qm9IsomeresMd(Dataset):
    def __init__(
        self,
        data_directory: str,
        size: int = 500,
        val: float = 0.1,
        test: float = 0.1,
    ) -> None:
        super().__init__(data_directory, "qm9_isomeres_md", None, size, val, test)

    @cached_property
    def schemes(self) -> list[str]:
        # huckel and mod_huckel won't work since we don't have ECPs
        return ["1e", "vsap", "sap", "minao"]

    @cached_property
    def elements(self) -> list[str]:
        return ["H", "C", "O"]

    def molecule(self, key: int) -> Mole:
        name = self.names[key]
        xyz = str(Path(self.xyz).joinpath(f"{name}.xyz"))

        return M(  # MD at  500 K: we assume molecules remain non-charged & closed-shell
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
