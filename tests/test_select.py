from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from rdsl.select import select_atom_ids, select_molecule
from rdsl.select.consts import SIDECHAIN_ATOMS


@pytest.mark.parametrize(
    "smiles, expr, expected_indices",
    [
        # Bymolecule
        ("C.O", "bymolecule elem C or elem O", [0, 1]),
        ("C.O", "bymolecule (elem C or elem O)", [0, 1]),
        ("N.CO", "(bymolecule elem N) or elem O", [0, 2]),
        # Byring
        ("c1ccccc1C", "byring (elem C and not ring)", [0, 1, 2, 3, 4, 5]),
        ("c1ccccc1C", "byring index 0", [0, 1, 2, 3, 4, 5]),
        ("c1ccccc1-c2ccccc2", "byring index 0", [0, 1, 2, 3, 4, 5]),
        ("c1ccccc1-c2ccccc2", "byring index 5", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        # byfunctional
        ("CC(=O)OC", "byfunctional index 1", [0, 1, 2, 3]),
        ("c1ccccc1C", "byfunctional index 6", [0, 1, 2, 3, 4, 5, 6]),
        # Index
        ("C.O", "index 0", [0]),
        ("C.O", "index 0 or index 1", [0, 1]),
        ("C.O", "index 0 and index 1", []),
        ("C.O.N", "index 0 or (index 1 and elem O)", [0, 1]),
        ("CCO.c1ccccc1.C(=O)O", "elem C and aromatic and index 3", [3]),
        ("CCO.c1ccccc1.C(=O)O", "bymolecule (elem C and aromatic and index 3)", [3, 4, 5, 6, 7, 8]),
        ("CCO.c1ccccc1.C(=O)O", "all in (bymolecule (elem C and aromatic and index 3))", [3, 4, 5, 6, 7, 8]),
        ("CCO.c1ccccc1.C(=O)O", "elem C in (bymolecule (index 0))", [0, 1]),
        ("CCO.c1ccccc1.C(=O)O", "elem C in (bymolecule index 0)", [0, 1]),
        ("CCO.c1ccccc1.C(=O)O", "elem C in bymolecule (index 0)", [0, 1]),
        ("CCO.c1ccccc1.C(=O)O", "elem C in bymolecule index 0", [0, 1]),
        # Flags
        ("NC(=O)Cc1ccccc1", "all", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "all in all", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "none", []),
        ("NC(=O)Cc1ccccc1", "aromatic", [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "aliphatic", [0, 1, 2, 3]),  # rest atoms
        ("NC(=O)Cc1ccccc1", "ring", [4, 5, 6, 7, 8, 9]),
        ("[N+]=C", "formal_charge 1", [0]),
        ("[N+]=C", "formal_charge > 0", [0]),
        ("[N+]=C", "partial_charge > 0.4", [0]),
        # Attributes
        ("NC(=O)Cc1ccccc1", "elem C", [1, 3, 4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "elem N", [0]),
        ("NC(=O)Cc1ccccc1", "elem O", [2]),
        ("NC(=O)Cc1ccccc1", "elem C+N", [0, 1, 3, 4, 5, 6, 7, 8, 9]),
        # Logic
        ("NC(=O)Cc1ccccc1", "not aromatic", [0, 1, 2, 3]),
        ("NC(=O)Cc1ccccc1", "aromatic and (elem C)", [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "aromatic and (elem C or elem N)", [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "aromatic and elem C or (elem N and not aromatic)", [0, 4, 5, 6, 7, 8, 9]),
        # SMARTS
        ("NC(=O)Cc1ccccc1", 'smarts "c1ccccc1"', [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "smarts 'c1ccccc1'", [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", 'smarts "C(=O)N"', [0, 1, 2]),
        # Quoted identifiers
        ("NC(=O)Cc1ccccc1", "elem 'C'", [1, 3, 4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", 'elem "N"', [0]),
        # Functional group
        ("CC(=O)C", "functional KeTone", [1, 2]),
        ("CC(=O)C", "functional 'KeTone'", [1, 2]),
        # Rings
        ("NC(=O)Cc1ccccc1", "ringsize 6", [4, 5, 6, 7, 8, 9]),
        ("NC(=O)Cc1ccccc1", "ringsize 5", []),
        ("NC(=O)Cc1ccccc1", "ringsize 5+6", [4, 5, 6, 7, 8, 9]),
        ("N1C=CC=C1", "ringsize 5", [0, 1, 2, 3, 4]),
        ("N1C=CC=C1", "ringsize 6", []),
        # RDKit Features definitions
        # I've checked that these replicate MolChemicalFeatureFactory using BaseFeatures.fdef
        ("CC(=O)O", "donors", [3]),
        ("CC(=O)O", "acceptors", [2, 3]),
        ("CC(=O)O", "neg_ionizable", [1, 2, 3]),
        ("CCN", "pos_ionizable", [2]),
        ("CCCC", "hydrophobes", [1, 2]),
        ("CC(C)C", "lumped_hydrophobes", [0, 1, 2, 3]),
        ("CS", "zn_binders", [0, 1]),
        # Extend
        ("NC(=O)Cc1ccccc1", "elem N extend 1", [0, 1]),
        ("NC(=O)Cc1ccccc1", "elem N extend 2", [0, 1, 2, 3]),
        ("NC(=O)Cc1ccccc1", "bound_to elem N", [0, 1]),
        ("NC(=O)Cc1ccccc1", "bt. elem N", [0, 1]),
        ("NC(=O)Cc1ccccc1", "neighbor elem N", [1]),
        ("NC(=O)Cc1ccccc1", "nb. elem N", [1]),
        # First / last
        ("NC(=O)Cc1ccccc1", "first elem C", [1]),
        ("NC(=O)Cc1ccccc1", "last elem C", [9]),
        ("NC(=O)Cc1ccccc1", "first ring", [4]),
        ("NC(=O)Cc1ccccc1", "last elem N", [0]),
        ("NC(=O)Cc1ccccc1", "first none", []),
        # Metals
        ("[Fe].C", "metals", [0]),
        ("C=CC(C)=O", "alerts", [0, 1, 2, 4]),
        ("C[C@H](O)C(=O)O", "stereo S", [1]),
        ("C[C@@H](O)C(=O)O", "stereo R", [1]),
    ],
)
def test_select(smiles, expr, expected_indices):
    mol = Chem.MolFromSmiles(smiles)
    selected = select_atom_ids(mol, expr)
    expected = list(expected_indices)

    np.testing.assert_array_equal(sorted(selected), sorted(expected))


def test_select_dist():
    # Select Oxygen (8)
    # Select N (9)

    mol = Chem.MolFromSmiles("NC(=O)Cc1ccccc1")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, randomSeed=42)

    # Verify indices
    o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")

    conf = mol.GetConformer()
    dist = conf.GetAtomPosition(o_idx).Distance(conf.GetAtomPosition(n_idx))
    print(f"Distance between O({o_idx}) and N({n_idx}): {dist}")

    # We expect distance > 1.0
    assert dist > 1.0, f"Distance {dist} is too small!"

    sel = select_atom_ids(mol, f"elem O within {dist + 0.5} of elem N")
    assert o_idx in sel

    sel_far = select_atom_ids(mol, f"elem O within {dist - 0.5} of elem N")
    assert o_idx not in sel_far

    sel_beyond = select_atom_ids(mol, f"elem O beyond {dist - 0.5} of elem N")
    assert o_idx in sel_beyond


def test_select_dist_no_unit():
    """Distance is always in Angstroms (no unit keyword)."""
    mol = Chem.MolFromSmiles("NC(=O)Cc1ccccc1")
    # mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()
    o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    real_dist = conf.GetAtomPosition(o_idx).Distance(conf.GetAtomPosition(n_idx))

    sel = select_atom_ids(mol, f"elem O within {real_dist + 0.5} of elem N")
    assert o_idx in sel

    sel_fail = select_atom_ids(mol, f"elem O within {real_dist - 0.5} of elem N")
    assert o_idx not in sel_fail


def test_select_around():
    """Test 'selection around dist'."""
    mol = Chem.MolFromSmiles("NC(=O)Cc1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()

    n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
    o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
    dist = conf.GetAtomPosition(n_idx).Distance(conf.GetAtomPosition(o_idx))

    # All atoms within dist + 0.5 of N. O should be in there.
    sel = select_atom_ids(mol, f"elem N around {dist + 0.5}")
    assert o_idx in sel
    assert n_idx in sel  # S1 should be in S1 around 12.3

    # Strictly less than dist
    sel_far = select_atom_ids(mol, f"elem N around {dist - 0.1}")
    assert o_idx not in sel_far
    assert n_idx in sel_far


def test_select_gap():
    """Test 'selection gap threshold'."""
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()

    pt = Chem.GetPeriodicTable()
    c1_idx = 0
    o_idx = 2

    d_c1_o = conf.GetAtomPosition(c1_idx).Distance(conf.GetAtomPosition(o_idx))
    r_c = pt.GetRvdw(6)
    r_o = pt.GetRvdw(8)
    real_gap = d_c1_o - r_c - r_o

    # Selection should include oxygen if threshold is less than or equal to real_gap
    # (assuming we are 'separated by a minimum of' threshold)
    sel = select_atom_ids(mol, f"index 0 gap {real_gap - 0.1}")
    assert o_idx in sel

    # Should NOT include oxygen if threshold is greater than real_gap
    sel_fail = select_atom_ids(mol, f"index 0 gap {real_gap + 0.1}")
    assert o_idx not in sel_fail


@pytest.fixture
def pdb_mol():
    """Fixture to load 7rpz.pdb."""
    pdb_path = Path(__file__).parent / "data" / "7rpz.pdb"
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    assert mol is not None
    return mol


def test_select_pdb_basic_ops(pdb_mol):
    """Test basic _PDB_OPS (chain, resn, resi, id, name, altloc, b, q)."""
    submol = select_molecule(pdb_mol, "chain A").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetChainId() == "A"

    submol = select_molecule(pdb_mol, "resn THR").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetResidueName() == "THR"

    submol = select_molecule(pdb_mol, "resv 1").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetResidueNumber() == 1

    submol = select_molecule(pdb_mol, "resi 1").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetResidueNumber() == 1

    submol = select_molecule(pdb_mol, "name CA").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetName().strip() == "CA"

    submol = select_molecule(pdb_mol, "alt A").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetAltLoc() == "A"

    submol = select_molecule(pdb_mol, "b > 24.84").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetTempFactor() > 24.84

    submol = select_molecule(pdb_mol, "q < 0.9").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetOccupancy() < 0.9

    submol = select_molecule(pdb_mol, "id 1").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetSerialNumber() == 1


def test_select_pdb_logic_ops(pdb_mol):
    """Test logical combinations of PDB ops."""
    submol = select_molecule(pdb_mol, "resn ALA and chain A").mol
    for atom in submol.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        assert residue_info.GetResidueName() == "ALA"
        assert residue_info.GetChainId() == "A"


def test_select_pdb_structural_flags(pdb_mol):
    """Test sidechain, backbone, and solvent flags."""

    side_chain_atom_names = set()
    for _resn, atom_names in SIDECHAIN_ATOMS.items():
        side_chain_atom_names.update(atom_names)

    submol = select_molecule(pdb_mol, "sidechain").mol
    for atom in submol.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        assert residue_info.GetResidueName().strip() in SIDECHAIN_ATOMS
        assert residue_info.GetName().strip() in side_chain_atom_names

    submol = select_molecule(pdb_mol, "backbone").mol
    for atom in submol.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        assert residue_info.GetResidueName().strip() in SIDECHAIN_ATOMS
        assert residue_info.GetName().strip() in ["N", "CA", "C", "O", "OXT"]

    submol = select_molecule(pdb_mol, "solvent").mol
    for atom in submol.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        assert residue_info.GetResidueName() == "HOH"
        assert residue_info.GetName().strip() in ["O", "H1", "H2"]


def test_select_pdb_molecule_flags(pdb_mol):
    """Test polymer, metals, protein, nucleic, and organic flags."""

    submol = select_molecule(pdb_mol, "polymer").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetResidueName() in SIDECHAIN_ATOMS

    submol = select_molecule(pdb_mol, "metals").mol
    for atom in submol.GetAtoms():
        residue_info = atom.GetPDBResidueInfo()
        assert residue_info.GetResidueName().strip() == "MG"
        assert residue_info.GetName().strip() == "MG"

    submol = select_molecule(pdb_mol, "protein").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetResidueName() in SIDECHAIN_ATOMS

    submol = select_molecule(pdb_mol, "nucleic").mol
    assert not submol

    submol = select_molecule(pdb_mol, "organic").mol
    organic_resnames = {atom.GetPDBResidueInfo().GetResidueName().strip() for atom in submol.GetAtoms()}
    assert "6IC" in organic_resnames
    assert "GDP" in organic_resnames
    # Ensure protein is not in organic
    assert "ALA" not in organic_resnames

    submol = select_molecule(pdb_mol, "inorganic").mol
    inorganic_resnames = {atom.GetPDBResidueInfo().GetResidueName().strip() for atom in submol.GetAtoms()}
    assert "MG" in inorganic_resnames
    # Even if MG is bonded to protein, the protein atoms should not be marked inorganic
    assert "SER" not in inorganic_resnames
    assert "ALA" not in inorganic_resnames
    # And specifically for 7rpz.pdb, it should just be the 1 MG atom (when removeHs=False, maybe 1? No, Mg has no Hs)
    assert submol.GetNumAtoms() == 1


def test_select_pdb_expansion(pdb_mol):
    """Test bychain and byres expansion on PDB molecule."""
    # resn 6IC is in chain A
    submol = select_molecule(pdb_mol, "bychain resn 6IC").mol
    for atom in submol.GetAtoms():
        assert atom.GetPDBResidueInfo().GetChainId() == "A"

    # Select all atoms in residues that have at least one CA atom
    # (should still be everything in the residues)
    # Let's try something more specific: residues containing Fluorine (elem F)
    # The organic ligand 6IC has Fluorines
    submol_f = select_molecule(pdb_mol, "byres elem F").mol
    # All atoms in the 6IC residue should be selected
    resnames = {a.GetPDBResidueInfo().GetResidueName().strip() for a in submol_f.GetAtoms()}
    assert "6IC" in resnames

    # And check that it actually selected more than just the F atoms
    f_atoms = [a for a in submol_f.GetAtoms() if a.GetSymbol() == "F"]
    non_f_atoms = [a for a in submol_f.GetAtoms() if a.GetSymbol() != "F"]
    assert len(f_atoms) > 0
    assert len(non_f_atoms) > 0

    submol_f_alias = select_molecule(pdb_mol, "byresidue elem F").mol
    assert submol_f_alias.GetNumAtoms() == submol_f.GetNumAtoms()
