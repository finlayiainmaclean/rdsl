import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem

from rdsl.select.consts import (
    _ARTIFACT_RESIDUES,
    _BACKBONE_ATOMS,
    _INORGANIC_RESIDUES,
    _NUCLEIC_ACID_RESIDUES,
    _PROTEIN_RESIDUES,
    _WATER_RESIDUES,
    SIDECHAIN_ATOMS,
)


def _is_protein(residue_info) -> bool:
    """Return True if this atom is a protein atom based on PDB residue info."""
    if residue_info is None:
        return False
    resn = residue_info.GetResidueName().strip()
    return resn in SIDECHAIN_ATOMS


def _is_nucleic_acid(residue_info) -> bool:
    """Return True if this atom is a nucleic acid atom based on PDB residue info."""
    if residue_info is None:
        return False
    resn = residue_info.GetResidueName().strip()
    return resn in _NUCLEIC_ACID_RESIDUES


def _is_polymer(residue_info) -> bool:
    """Return True if this atom is a polymer atom based on PDB residue info."""
    if residue_info is None:
        return False
    resn = residue_info.GetResidueName().strip()
    return resn in _NUCLEIC_ACID_RESIDUES or resn in SIDECHAIN_ATOMS


def _is_water(residue_info) -> bool:
    """Return True if this atom is a water atom based on PDB residue info."""
    if residue_info is None:
        return False
    return residue_info.GetResidueName().strip() in _WATER_RESIDUES


def _is_sidechain(residue_info) -> bool:
    """Return True if this atom is a sidechain atom based on PDB residue info."""
    if residue_info is None:
        return False

    resn = residue_info.GetResidueName().strip()
    name = residue_info.GetName().strip()

    # Must be a known amino acid
    if resn not in SIDECHAIN_ATOMS:
        return False

    return name in SIDECHAIN_ATOMS[resn]


def _is_artifact(residue_info) -> bool:
    """Return True if this atom is an artifact atom based on PDB residue info."""
    if residue_info is None:
        return False
    resn = residue_info.GetResidueName().strip()
    return resn in _ARTIFACT_RESIDUES


def _is_backbone(residue_info) -> bool:
    """Return True if this atom is a backbone atom based on PDB residue info."""
    if residue_info is None:
        return False

    resn = residue_info.GetResidueName().strip()
    # Must be a known amino acid
    if resn not in SIDECHAIN_ATOMS:
        return False
    name = residue_info.GetName().strip()
    return name in _BACKBONE_ATOMS


def _get_inorganic_atoms(mol: Chem.Mol) -> np.ndarray:
    """Return boolean mask of atoms belonging to inorganic fragments or residues."""
    frags = Chem.GetMolFrags(mol, asMols=False)
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)

    for frag_atoms in frags:
        resnames = set()
        has_carbon = False
        for i in frag_atoms:
            atom = mol.GetAtomWithIdx(int(i))
            if atom.GetAtomicNum() == 6:
                has_carbon = True
            info = atom.GetPDBResidueInfo()
            if info is not None:
                resnames.add(info.GetResidueName().strip())

        # If it's a mixed fragment (inorganic residues + carbon/protein/etc.),
        # only mark the specific inorganic residues.
        if (resnames & _INORGANIC_RESIDUES) and (has_carbon or (resnames - _INORGANIC_RESIDUES - _WATER_RESIDUES)):
            for i in frag_atoms:
                atom = mol.GetAtomWithIdx(int(i))
                info = atom.GetPDBResidueInfo()
                if info and info.GetResidueName().strip() in _INORGANIC_RESIDUES:
                    mask[int(i)] = True
            continue

        # For purely inorganic fragments (known inorganic residue OR no carbon at all)
        if resnames & _INORGANIC_RESIDUES:
            mask[list(frag_atoms)] = True
            continue

        if not has_carbon and not (resnames & _WATER_RESIDUES):
            mask[list(frag_atoms)] = True

    return mask


def _get_organic_atoms(mol: Chem.Mol, inorganic_mask: np.ndarray) -> np.ndarray:
    """Return boolean mask of atoms belonging to organic fragments or residues."""
    frags = Chem.GetMolFrags(mol, asMols=False)
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)

    for frag_atoms in frags:
        resnames = set()
        has_carbon = False
        for i in frag_atoms:
            atom = mol.GetAtomWithIdx(int(i))
            if atom.GetAtomicNum() == 6:
                has_carbon = True
            info = atom.GetPDBResidueInfo()
            if info is not None:
                resnames.add(info.GetResidueName().strip())

        if not has_carbon:
            continue

        # If it's a mixed fragment (bonded to protein/nucleic/inorganic),
        # only mark atoms that don't belong to those other categories.
        other_categories = _PROTEIN_RESIDUES | _NUCLEIC_ACID_RESIDUES | _WATER_RESIDUES | _ARTIFACT_RESIDUES
        if resnames & other_categories:
            for i in frag_atoms:
                if inorganic_mask[int(i)]:
                    continue
                atom = mol.GetAtomWithIdx(int(i))
                info = atom.GetPDBResidueInfo()
                if info:
                    resn = info.GetResidueName().strip()
                    if resn in other_categories:
                        continue
                mask[int(i)] = True
            continue

        # Purely organic fragment (no protein/nucleic etc.)
        # Still respect the inorganic_mask for specific atoms (e.g. metals in ligands)
        for i in frag_atoms:
            if not inorganic_mask[int(i)]:
                mask[int(i)] = True

    return mask


def _is_metal(atom: rdchem.Atom) -> bool:
    return atom.GetSymbol() in {
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Be",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Al",
        "Ga",
        "In",
        "Sn",
        "Pb",
        "Bi",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    }


def _create_context(mol: Chem.Mol) -> pd.DataFrame:
    """Create a DataFrame with atom properties for selection."""
    data = []
    inorganic_mask = _get_inorganic_atoms(mol)
    organic_mask = _get_organic_atoms(mol, inorganic_mask)

    if not mol.GetAtomWithIdx(0).HasProp("_GasteigerCharge"):
        AllChem.ComputeGasteigerCharges(mol)

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    for idx, atom in enumerate(mol.GetAtoms()):
        residue_info = atom.GetPDBResidueInfo()
        data.append({
            "index": atom.GetIdx(),
            "elem": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "mass": atom.GetMass(),
            "atom_map_number": atom.GetAtomMapNum(),
            "valence": atom.GetTotalValence(),
            "explicit_valence": atom.GetValence(Chem.ValenceType.EXPLICIT),
            "implicit_valence": atom.GetValence(Chem.ValenceType.IMPLICIT),
            "isotope": atom.GetIsotope(),
            "aromatic": atom.GetIsAromatic(),
            "aliphatic": not atom.GetIsAromatic(),
            "hetatm": residue_info.GetIsHeteroAtom() if residue_info else None,
            "hydrogens": atom.GetSymbol() == "H",
            "heavy": atom.GetSymbol() != "H",
            "ring": atom.IsInRing(),
            "degree": atom.GetDegree(),
            "formal_charge": atom.GetFormalCharge(),
            "partial_charge": (atom.GetDoubleProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else None),
            "radical_electrons": atom.GetNumRadicalElectrons(),
            "hybridization": str(atom.GetHybridization()),
            "stereo": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
            "chain": residue_info.GetChainId() if residue_info else None,
            "residue": (
                f"{residue_info.GetChainId()}:{residue_info.GetResidueNumber()}{residue_info.GetInsertionCode().strip()}"
                if residue_info
                else None
            ),
            "resi": (
                f"{residue_info.GetResidueNumber()}{residue_info.GetInsertionCode().strip()}" if residue_info else None
            ),
            "resn": residue_info.GetResidueName().strip() if residue_info else None,
            "resv": residue_info.GetResidueNumber() if residue_info else None,
            "name": residue_info.GetName().strip() if residue_info else None,
            "alt": residue_info.GetAltLoc() if residue_info else None,
            "b": residue_info.GetTempFactor() if residue_info else None,
            "q": residue_info.GetOccupancy() if residue_info else None,
            "id": residue_info.GetSerialNumber() if residue_info else None,
            "sidechain": _is_sidechain(residue_info),
            "backbone": _is_backbone(residue_info),
            "solvent": _is_water(residue_info),
            "polymer": _is_polymer(residue_info),
            "protein": _is_protein(residue_info),
            "nucleic": _is_nucleic_acid(residue_info),
            "inorganic": bool(inorganic_mask[idx]),
            "organic": bool(organic_mask[idx]),
            "metals": _is_metal(atom),
        })

    return pd.DataFrame(data)
