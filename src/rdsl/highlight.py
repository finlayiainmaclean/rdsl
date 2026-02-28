from collections.abc import Iterable

from rdkit import Chem


def highlight_atoms(mol: Chem.Mol, /, *, atom_ids: Iterable[int]) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.__sssAtoms = [int(i) for i in atom_ids]
    mol.__sssQry = None
    return mol
