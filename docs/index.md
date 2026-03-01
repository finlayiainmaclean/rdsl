# RDSL (RDKit Selection Language)

[![Release](https://img.shields.io/github/v/release/finlayiainmaclean/rdsl)](https://img.shields.io/github/v/release/finlayiainmaclean/rdsl)
[![Build status](https://img.shields.io/github/actions/workflow/status/finlayiainmaclean/rdsl/main.yml?branch=main)](https://github.com/finlayiainmaclean/rdsl/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/finlayiainmaclean/rdsl)](https://img.shields.io/github/license/finlayiainmaclean/rdsl)

RDKit Selection Language (RDSL) is a domain-specific language for selecting atoms and substructures within RDKit molecules, inspired by languages like PyMOL but tailored for chemical informatics.

## Installation

```bash
pip install rdsl
```

## Quick Start

```python
from rdkit import Chem
from rdsl import select_atom_ids, select_molecule

mol = Chem.MolFromSmiles("N1C=C(F)C=C1C1=CC2NCCC(C(=O)[O-])C=2C=C1")

# Select atom indices
atom_ids = select_atom_ids(mol, "elem C and ring")

# Extract a subset molecule
result = select_molecule(mol, "aromatic and beyond 3.0 of donors")
submol = result.mol
```

## Selection Language

RDSL provides a powerful syntax for defining atom selections.

### Basic Selectors
*   `all` (also `*`) : Select all atoms
*   `none` : Select no atoms
*   `hydrogens` (also `h.`) : Select hydrogen atoms
*   `heavy` : Select heavy (non-hydrogen) atoms

### Operators
*   **Indices**:
    *   `index <range>` : Select atoms by index range (e.g., `index 1+3-5`)
*   **Atomic Properties**:
    *   `atomic_number <int>` : Select atoms by atomic number
    *   `valence <int>` : Select atoms by valence
    *   `explicit_valence <int>` : Select atoms by explicit valence
    *   `implicit_valence <int>` : Select atoms by implicit valence
    *   `elem <symbol>` (also `e.`) : Select atoms by element symbol
    *   `mass <float>` : Select atoms by mass
    *   `degree <int>` : Select atoms by degree
    *   `formal_charge <int>` (also `fc.`) : Select atoms by formal charge
    *   `partial_charge <float>` (also `pc.`) : Select atoms by partial charge
    *   `isotope <int>` : Select atoms by isotope
    *   `atom_map_number <int>` : Select atoms by atom map number
    *   `radical_electrons <int>` : Select atoms by number of radical electrons
    *   `hybridization <OTHER/S/SP/SP2/SP2D/SP3/SP3D/SP3D2/UNSPECIFIED>` : Select atoms by hybridization
    *   `stereo <R/S>` : Select atoms by R/S stereochemistry
    *   `ring` (also `cyclic`) : Select atoms in rings
    *   `acyclic` : Select atoms not in rings
    *   `aromatic` : Select aromatic atoms
    *   `aliphatic` : Select aliphatic atoms
*   **Pharmacophores**:
    *   `donors` (also `don.`) : Hydrogen bond donors
    *   `acceptors` (also `acc.`) : Hydrogen bond acceptors
    *   `hydrophobes` : Hydrophobic atoms
    *   `lumped_hydrophobes` : Lumped hydrophobic atoms
    *   `pos_ionizable` : Positively ionizable atoms
    *   `neg_ionizable` : Negatively ionizable atoms
    *   `zn_binders` : Zinc binding atoms
*   **MedChem Alerts**:
    *   `alerts` : Alerts from the PAINS, BMS and Brenk
*   **SMARTS** :
    *   `smarts "<pattern>"` : Custom SMARTS patterns
*   **Logical**:
    *   `and` (also `&`) : Logical AND
    *   `or` (also `|`) : Logical OR
    *   `not` (also `!`) : Logical NOT
    *   `<selection> in <selection>` : Filter atoms in the left selection that are also in the right selection
*   **Proximity**:
    *   `<selection> around <radius>` (also `a.`) : Atoms within radius (Angstroms) of any atom in the selection
    *   `<selection_a> within <distance> of <selection_b>` (also `w.`) : Atoms in selection A that are within distance of any atom in selection B
    *   `<selection_a> beyond <distance> of <selection_b>` (also `be.`) : Opposite of `within`
    *   `<selection> gap <radius>` : Atoms where VDW surfaces are within radius of any atom in the selection
*   **Bond Expansion**:
    *   `<selection> extend <n>` (also `xt.`) : Expand selection by `n` bonds (includes intermediate atoms)
*   **Object Expansion**:
    *   `byring` : Selects any ring system connected to the current selection
    *   `byres` (also `br.`) : Selects any residue connected to the current selection
    *   `bymolecule` (also `bm.`) : Selects any molecule connected to the current selection
    *   `bychain` (also `bc.`) : Selects any chain connected to the current selection
    *   `byfunctional` : Selects the largest functional group connected to the current selection
*   **Positional**:
    *   `first <selection>` : The first atom in the selection
    *   `last <selection>` : The last atom in the selection
    *   `nth <range> <selection>` : Selected atoms by position in the selection (e.g., `nth 1+3 aromatic`)
*   **Topology**:
    *   `neighbor <selection>` (also `nbr.`) : Neighbors of the current selection
    *   `bound_to <selection>` (also `bto.`) : Atoms bound to the current selection
*   **Rings**:
    *   `ringsize <n>` : Atoms in rings of size n
    *   `inring` : Atoms in rings
*    **Functional Groups**:
    *   `functional "<name>"` : Select atoms in specific functional group
*    **PDB**:
    *   `resi <n>` (also `i.`) : Select atoms by residue index
    *   `resn <name>` (also `r.`) : Select atoms by residue name
    *   `resv <id>` : Select atoms by residue variant
    *   `chain <name>` (also `c.`) : Select atoms by chain
    *   `name <name>` (also `n.`) : Select atoms by atom name
    *   `alt <name>` : Select atoms by alternate location
    *   `id <id>` : Select atoms by atom id
    *   `b <value>` : Select atoms by b-factor
    *   `q <value>` : Select atoms by occupancy
    *   `hetatm` : Select heteroatoms
    *   `protein` : Select protein atoms
    *   `nucleic` : Select nucleic atoms
    *   `polymer` (also `pol.`) : Select polymer atoms
    *   `solvent` (also `sol.`) : Select solvent atoms
    *   `sidechain` (also `sc.`) : Select sidechain atoms
    *   `backbone` (also `bb.`) : Select backbone atoms
    *   `metals` : Select metal atoms
    *   `organic` (also `org.`) : Select organic atoms
    *   `inorganic` (also `ino.`) : Select inorganic atoms
    *   `artifact` : Select artifact atoms

## Functional Groups

RDSL includes a built-in library of functional group patterns with hierarchy support. This uses the patterns from [SmartChemist](https://github.com/torbengutermuth/SmartChemist) and carry a Creative Commons Attribution-NoDerivatives 4.0 International Public License.

```python
from rdsl import get_functional_group_matches

# Get all matches as a pandas DataFrame
df = get_functional_group_matches(mol, include_overshadowed=False)
print(df[['name', 'atom_ids']])
```

### Hierarchy and Overshadowing

The functional group definitions include hierarchical relationships. By default, `include_overshadowed=False` will filter out smaller groups that are contained within larger ones (e.g., 'methyl' will be excluded if 'toluene' is found).
