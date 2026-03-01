import abc

# Setup recursion limit and pyparsing packrat
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pyparsing import DelimitedList, ParserElement, QuotedString, Regex, Word, alphanums
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from .predefined import PREDEFINED_SMARTS

sys.setrecursionlimit(3000)
ParserElement.enable_packrat()

_FEATURE_FACTORY = cast(Any, ChemicalFeatures).BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))

_FEATURE_MAP = {
    "donors": "Donor",
    "acceptors": "Acceptor",
    "pos_ionizable": "PosIonizable",
    "neg_ionizable": "NegIonizable",
    "hydrophobes": "Hydrophobe",
    "lumped_hydrophobes": "LumpedHydrophobe",
    "zn_binders": "ZnBinder",
}


def _parse_range(tokens) -> list[int]:
    """Parse a range of the form 'start-end'."""
    ranges = []
    for token in tokens:
        if "-" in token:
            start, end = map(int, token.split("-"))
            ranges.extend(range(start, end + 1))
        else:
            ranges.append(int(token))
    return ranges


_INTEGER = DelimitedList(Regex(r"\d+(-\d+)?").set_parse_action(_parse_range), delim="+")
_IDENTIFIER_BASE = (
    Word(alphanums, alphanums + "*") | QuotedString('"', unquote_results=True) | QuotedString("'", unquote_results=True)
)
_IDENTIFIER = DelimitedList(_IDENTIFIER_BASE, delim="+")
_DECIMAL_REGEX = r"[+-]?\d*\.?\d*"
_DECIMAL = Regex(_DECIMAL_REGEX).set_parse_action(lambda t: float(t[0]))

# Dynamically generate flag ops from SMARTS
_SMARTS_FLAG_OPS: list[tuple[str, list[str]]] = [(name, []) for name in PREDEFINED_SMARTS]
_FEATURE_FLAG_OPS: list[tuple[str, list[str]]] = [(name, []) for name in _FEATURE_MAP]

_FLAG_OPS = [
    ("all", ["*"]),
    ("none", []),
    ("aromatic", []),
    ("aliphatic", []),
    ("hetatm", []),
    ("hydrogens", ["h.", "hydrogen"]),
    ("heavy", []),
    ("cyclic", ["ring"]),
    ("acyclic", []),
    ("sidechain", ["sc."]),
    ("backbone", ["bb."]),
    ("solvent", ["sol."]),
    ("metals", ["metal"]),
    ("polymer", ["pol."]),
    ("protein", []),
    ("nucleic", []),
    ("inorganic", ["ino."]),
    ("organic", ["org."]),
    ("artifact", []),
    ("donors", ["don.", "donor"]),
    ("acceptors", ["acc.", "acceptor"]),
    ("artefact", []),
    *_SMARTS_FLAG_OPS,
    *[(name, []) for name in _FEATURE_MAP if name not in ("donors", "acceptors")],
]

_ATTR_OPS = [
    ("elem", ["e."], _IDENTIFIER),
    ("atomic_number", [], _INTEGER),
    ("isotope", [], _INTEGER),
    ("atom_map_number", [], _INTEGER),
    ("mass", [], _DECIMAL),
    ("explicit_valence", [], _DECIMAL),
    ("implicit_valence", [], _DECIMAL),
    ("valence", [], _DECIMAL),
    ("index", ["idx."], _INTEGER),
    ("degree", [], _INTEGER),
    ("formal_charge", ["fc."], _INTEGER),
    ("partial_charge", ["pc."], _DECIMAL),
    ("radical_electrons", [], _INTEGER),
    ("stereo", [], _IDENTIFIER),
    ("hybridization", [], _IDENTIFIER),
]

_PDB_OPS = [
    ("chain", ["c."], _IDENTIFIER),
    ("resn", ["r."], _IDENTIFIER),
    ("resi", ["i."], _IDENTIFIER),
    ("resv", [], _INTEGER),
    ("name", ["n."], _IDENTIFIER),
    ("alt", [], _IDENTIFIER),
    ("b", [], _DECIMAL),
    ("q", [], _DECIMAL),
    ("id", [], _INTEGER),
]
_ATTR_OPS += _PDB_OPS

_SMARTS_OP = [("smarts", [])]
_FUNCTIONAL_OP = [("functional", [])]
_RING_OPS = [
    ("inring", []),
    ("ringsize", []),
]
_DIST_OPS = [
    ("within", ["w."]),
    ("beyond", ["be."]),
]
_UNARY_OPS = [("not", ["!"])]
_BINARY_OPS = [("and", ["&"]), ("or", ["|"])]
_SUBSET_OPS = [("in", ["in"])]
_EXPANSION_OPS = [
    ("bymolecule", ["bm."]),
    ("bychain", ["bc."]),
    ("byres", ["br."]),
    ("byring", []),
    ("byfunctional", []),
]
_FIRSTLAST_OPS = [
    ("first", []),
    ("last", []),
]
_BOND_OPS = [("bound_to", ["bto."])]
_NEIGHBOR_OPS = [("neighbor", ["nbr."])]
_AROUND_OPS = [("around", ["a."])]
_GAP_OPS = [("gap", [])]
_EXTEND_OPS = [("extend", ["xt."])]
_NTH_OPS = [("nth", [])]

_ALL_OPS = (
    _FLAG_OPS
    + _ATTR_OPS
    + _SMARTS_OP
    + _FUNCTIONAL_OP
    + _RING_OPS
    + _EXPANSION_OPS
    + _FIRSTLAST_OPS
    + _NTH_OPS
    + _DIST_OPS
    + _UNARY_OPS
    + _BINARY_OPS
    + _SUBSET_OPS
    + _BOND_OPS
    + _NEIGHBOR_OPS
    + _AROUND_OPS
    + _GAP_OPS
    + _EXTEND_OPS
)


def _flatten(op: tuple[str, list[str]] | tuple[str, list[str], Any]) -> tuple[str, ...]:
    return op[0], *op[1]


_ALIASES = {alias: op[0] for op in _ALL_OPS for alias in _flatten(op)}


class SelectError(Exception):
    """Base class for exceptions in this module."""


class ConformerError(SelectError):
    """Raised when a 3D conformer is required but missing."""

    def __init__(self, msg="Molecule has no 3D conformer"):
        super().__init__(msg)


class SelectionParserError(SelectError):
    """Raised when parsing a selection expression fails."""

    def __init__(self, message=None, expected=None, got=None):
        if message:
            super().__init__(message)
        elif expected and got:
            super().__init__(f"Expected {expected}, got {got}")
        else:
            super().__init__("Selection parsing failed")


class BrokenBondError(SelectError):
    """Raised when a broken bond cannot be correctly handled."""

    def __init__(self, bond_idx: int):
        super().__init__(f"Could not find the removed atom in the bond {bond_idx}")


class InvalidPatternError(SelectError):
    """Raised when an invalid SMARTS or SMILES pattern is provided."""

    def __init__(self, message=None, pattern=None, pattern_type=None, kw=None):
        if message:
            super().__init__(message)
        elif pattern:
            super().__init__(f"Invalid SMARTS pattern: {pattern}")
        elif kw and pattern_type:
            super().__init__(f"Invalid pattern type for {kw}: {pattern_type}")
        else:
            super().__init__("Invalid chemical pattern")


class BaseOp(abc.ABC):
    @abc.abstractmethod
    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        """Apply the operation to the molecule."""
