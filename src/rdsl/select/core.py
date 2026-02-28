from typing import Literal, NamedTuple, Optional

import numpy as np
from loguru import logger
from rdkit import Chem

from rdsl.select.parser import _parse_expr
from rdsl.select.utils import _create_context

class SelectionResult(NamedTuple):
    """Result of a molecule selection.
    
    Attributes:
        mol: The extracted subset molecule, or None if no atoms were selected.
        atom_mapping: Mapping from new atom indices to original atom indices.
        bond_mapping: Mapping from new bond indices to original bond indices.
    """
    mol: Optional[Chem.Mol]
    atom_mapping: dict[int, int]
    bond_mapping: dict[int, int]


def select_atom_ids(mol: Chem.Mol, expr: str) -> np.ndarray:
    """Select atoms from an RDKit molecule.

    Args:
        mol: RDKit molecule
        expr: Selection expression

    Returns:
        Array of selected atom indices (0-based)

    Examples:
        >>> select(mol, "aromatic")  # All aromatic atoms
        >>> select(mol, "elem C and ring")  # Carbon atoms in rings
        >>> select(mol, 'smarts "c1ccccc1"')  # Benzene rings
        >>> select(mol, "ringsize 6")  # Atoms in 6-membered rings
        >>> select(mol, "elem C within 3.0 of elem N")  # Carbons within 3.0 Angstroms of nitrogen
        >>> select(mol, "first elem C")  # First carbon by index
        >>> select(mol, "last ring")  # Last atom in ring
    """
    parsed = _parse_expr(expr)
    ctx = _create_context(mol)
    selected = parsed[0].apply(mol, ctx)
    atom_ids = np.where(selected)[0]

    # Store selected atoms on molecule for jupyter visualisation
    mol.__sssAtoms = [int(i) for i in atom_ids]
    mol.__sssQry = None

    return atom_ids


def select_atoms(mol: Chem.Mol, expr: str) -> list[Chem.Atom]:
    """Select atoms from an RDKit molecule and return them as Atom objects.
    
    Args:
        mol: RDKit molecule
        expr: Selection expression

    Returns:
        List of selected atom objects

    Examples:
        >>> select_atoms(mol, "aromatic")  # All aromatic atoms
        >>> select_atoms(mol, "elem C and ring")  # Carbon atoms in rings
        >>> select_atoms(mol, 'smarts "c1ccccc1"')  # Custom SMARTS pattern
        >>> select_atoms(mol, "ringsize 6")  # Atoms in 6-membered rings
        >>> select_atoms(mol, "elem C within 3.0 of elem N")  # Carbons within 3.0 Angstroms of nitrogen
        >>> select_atoms(mol, "first elem C")  # First carbon by index
        >>> select_atoms(mol, "last ring")  # Last atom in ring
    """
    atom_ids = select_atom_ids(mol, expr)
    return [mol.GetAtomWithIdx(int(i)) for i in atom_ids]


def select_molecule(
    mol: Chem.Mol, 
    expr: str, 
    broken_bonds: Literal["radicals", "hydrogens", "wildcards"] = "hydrogens",
) -> SelectionResult:
    """Extract a subset of a molecule based on a selection expression.

    Args:
        mol: RDKit molecule
        expr: Selection expression
        broken_bonds: How to handle bonds broken between atoms in the subset and atoms outside.
            * `hydrogens`: Insert implicit hydrogens when bonds are broken, to maintain valence (default).
            * `radicals`: Break the bond and leave the atoms with an unsatisfied valence.
            * `wildcards`: Replace broken bonds with wildcard atoms (*).

    Returns:
        A SelectionResult object containing the subset molecule and mappings.

    Examples:
        >>> result = select_molecule(mol, "aromatic")
        >>> submol = result.mol
        >>> original_idx = result.atom_mapping[0]
    """
    selected_indices = select_atom_ids(mol, expr)

    if len(selected_indices) == 0:
        return SelectionResult(None, {}, {})

    # Get atoms to delete (everything NOT selected)
    all_indices = set(range(mol.GetNumAtoms()))
    to_delete = sorted(all_indices - set(selected_indices), reverse=True)

    # Copy molecule and handle broken bonds
    new_mol = Chem.RWMol(mol)
    
    # Store original indices in atom/bond properties for mapping later
    for atom in new_mol.GetAtoms():
        atom.SetIntProp("_original_idx", atom.GetIdx())
    for bond in new_mol.GetBonds():
        bond.SetIntProp("_original_idx", bond.GetIdx())
    
    if broken_bonds == "radicals":
        for atom in new_mol.GetAtoms():
            atom.SetNumExplicitHs(atom.GetTotalNumHs())
            atom.SetNoImplicit(True)

    elif broken_bonds == "wildcards":
        # Find which bonds connect a selected atom to a non-selected one
        selected_set = set(selected_indices)
        has_coords = mol.GetNumConformers() > 0
        
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            
            kept_idx = None
            removed_idx = None
            if idx1 in selected_set and idx2 not in selected_set:
                kept_idx = idx1
                removed_idx = idx2
            elif idx2 in selected_set and idx1 not in selected_set:
                kept_idx = idx2
                removed_idx = idx1
                
            if kept_idx is not None:
                # Add wildcard atom and connect with the same bond type
                wildcard_idx = new_mol.AddAtom(Chem.Atom(0)) # Atomic number 0 is '*'
                new_mol.AddBond(kept_idx, wildcard_idx, bond.GetBondType())
                
                # Set 3D position if conformers exist
                if has_coords:
                    for i in range(mol.GetNumConformers()):
                        pos = mol.GetConformer(i).GetAtomPosition(removed_idx)
                        new_mol.GetConformer(i).SetAtomPosition(wildcard_idx, pos)

    # Delete atoms in reverse order to keep indices valid
    for idx in to_delete:
        new_mol.RemoveAtom(int(idx))

    subset_mol = new_mol.GetMol()

    # Collect mapping and clear properties
    atom_mapping = {}
    bond_mapping = {}
   
    for atom in subset_mol.GetAtoms():
        if atom.HasProp("_original_idx"):
            atom_mapping[atom.GetIdx()] = atom.GetIntProp("_original_idx")
            atom.ClearProp("_original_idx")
    for bond in subset_mol.GetBonds():
        if bond.HasProp("_original_idx"):
            bond_mapping[bond.GetIdx()] = bond.GetIntProp("_original_idx")
            bond.ClearProp("_original_idx")

    # Sanitization
    sani_fail = Chem.SanitizeMol(subset_mol, catchErrors=True)
    if sani_fail:
        # Some subsets might not be sanitizable (e.g. radicals if broken_bonds="radicals")
        logger.debug("Sanitization produced warnings for subset molecule")

    return SelectionResult(subset_mol, atom_mapping, bond_mapping)
