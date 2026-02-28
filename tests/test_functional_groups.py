import pandas as pd
from rdkit import Chem

from rdsl.functional_groups import get_functional_group_matches


def test_basic_functional_groups():
    # Ethanol has a hydroxy group
    mol = Chem.MolFromSmiles("CCO")
    df = get_functional_group_matches(mol)
    assert isinstance(df, pd.DataFrame)
    assert "hydroxy" in df["name"].values

    # Check highlighting and mol copy
    row = df[df["name"] == "hydroxy"].iloc[0]
    assert row["atom_ids"] == (2,)
    assert hasattr(row["mol"], "__sssAtoms")
    assert row["mol"].__sssAtoms == [2]
    assert hasattr(row["mol"], "__sssBonds")
    assert row["mol"].__sssBonds == []  # No bonds for single atom match
    # Ensure it's a copy
    assert row["mol"] is not mol


def test_benzene_highlighting():
    # Benzene ring should have 6 atoms and 6 bonds highlighted
    mol = Chem.MolFromSmiles("c1ccccc1")
    df = get_functional_group_matches(mol)
    row = df[df["name"] == "benzene"].iloc[0]
    assert len(row["atom_ids"]) == 6
    assert hasattr(row["mol"], "__sssBonds")
    # A 6-membered ring has 6 bonds
    assert len(row["mol"].__sssBonds) == 6


def test_hierarchy_filtering_toluene():
    # Toluene has a methyl group and a benzene ring
    mol = Chem.MolFromSmiles("Cc1ccccc1")

    # Default: include_overshadowed=False
    df = get_functional_group_matches(mol)
    names = set(df["name"].values)
    assert "toluene" in names
    assert "benzene" in names
    assert "methyl" not in names


def test_include_overshadowed_toluene():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    df = get_functional_group_matches(mol, include_overshadowed=True)
    names = set(df["name"].values)

    assert "toluene" in names
    assert "benzene" in names
    assert "methyl" in names


def test_hierarchy_filtering_cresol():
    mol = Chem.MolFromSmiles("Cc1ccc(O)cc1")
    df = get_functional_group_matches(mol, include_overshadowed=False)
    names = set(df["name"].values)

    assert "p-cresol" in names
    assert "phenol" in names
    assert "benzene" in names
    assert "hydroxy" not in names
    assert "toluene" not in names
    assert "methyl" not in names


def test_empty_mol():
    mol = Chem.Mol()
    df = get_functional_group_matches(mol)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
