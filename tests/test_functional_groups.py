import pandas as pd
from rdkit import Chem

from rdsl.functional_groups import get_functional_group_matches


def test_basic_functional_groups():
    # Ethanol has a hydroxy group
    mol = Chem.MolFromSmiles("CCO")
    df = get_functional_group_matches(mol)
    assert isinstance(df, pd.DataFrame)
    assert "ethoxy" in df["name"].values
    row = df[df["name"] == "ethoxy"].iloc[0]
    assert row["atom_ids"] == (0,1,2,)


def test_benzene_highlighting():
    # Benzene ring should have 6 atoms and 6 bonds highlighted
    mol = Chem.MolFromSmiles("c1ccccc1")
    df = get_functional_group_matches(mol)
    row = df[df["name"] == "benzene"].iloc[0]
    assert len(row["atom_ids"]) == 6


def test_hierarchy_filtering_toluene():
    # Toluene has a methyl group and a benzene ring
    mol = Chem.MolFromSmiles("Cc1ccccc1")

    # Default: include_overshadowed=False
    df = get_functional_group_matches(mol, include_overshadowed=False)
    names = set(df["name"].values)
    assert "toluene" in names
    assert "benzene" not in names
    assert "methyl" not in names


def test_include_overshadowed_toluene():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    df = get_functional_group_matches(mol, include_overshadowed=True)
    names = set(df["name"].values)

    assert "toluene" in names
    assert "benzene" in names
    assert "methyl" in names



def test_empty_mol():
    mol = Chem.Mol()
    df = get_functional_group_matches(mol)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
