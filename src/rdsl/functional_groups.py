import ast
from pathlib import Path

import pandas as pd
from rdkit import Chem

# WARNING: These SMARTS patterns are taken from
# https://github.com/torbengutermuth/SmartChemist/blob/main/smarts/License_for_patterns_here
# These patterns have aCreative Commons Attribution-NoDerivatives 4.0 International Public License
# If you use these patterns in your own code, remember to credit them!
# Do NOT modify the patterns without permission!

# Load and prepare the patterns data from the local CSV file
# The CSV contains SMARTS and SMILES patterns categorized by functional group, biological relevance, or cyclicity
_csv_path = Path(__file__).parent / "data" / "smarts_with_hierarchy.csv"
_df = pd.read_csv(_csv_path, skiprows=1)
_df = _df.dropna(subset=["trivialname", "SMARTS"])
_df["trivialname"] = _df["trivialname"].str.lower()


def _parse_hierarchy(h_str):
    if pd.isna(h_str) or h_str == "" or h_str == "[]":
        return []
    try:
        # Hierarchy is stored as string representation of list "[15, 38]"
        return ast.literal_eval(h_str)
    except (ValueError, SyntaxError):
        return []


_df["hierarchy_indices"] = _df["Hierarchy"].apply(_parse_hierarchy)

# Map indices to trivial names for filtering
_index_to_name = dict(zip(_df.index, _df.trivialname, strict=False))

# Map trivial names to their parent names (overshadowing groups)
HIERARCHY_MAP: dict[str, set[str]] = {}
for _row in _df.reset_index().to_dict("records"):
    parents = {_index_to_name[idx] for idx in _row["hierarchy_indices"] if idx in _index_to_name}
    if parents:
        # Note: In the CSV, Hierarchy column lists groups that are PARENTS or COMPONENTS of the current group.
        # e.g., Toluene [102] where 102 is Methyl.
        # This means Methyl is "overshadowed" if Toluene is found.
        HIERARCHY_MAP[_row["trivialname"]] = parents

# Map the groups to their respective RDKit pattern types
_pattern_mapping = {
    "functional_group": "smarts",
    "biological": "smarts",
    "cyclic": "smiles",
}
_df["pattern_type"] = _df["group"].map(_pattern_mapping)

# Internal storage of compiled patterns with metadata
_FUNCTIONAL_GROUP_PATTERNS_INTERNAL: list[dict] = []
for _row in _df.reset_index().to_dict("records"):
    _name = _row["trivialname"]
    _pattern_str = _row["SMARTS"]
    _type = _row["pattern_type"]

    _pattern_mol = Chem.MolFromSmarts(_pattern_str) if _type == "smarts" else Chem.MolFromSmiles(_pattern_str)

    if _pattern_mol is not None:
        _FUNCTIONAL_GROUP_PATTERNS_INTERNAL.append({
            "name": _name,
            "pattern": _pattern_mol,
            "smarts": _pattern_str,
            "group": _row["group"],
            "pattern_type": _type,
            "priority": _row["index"],
            "hierarchy": HIERARCHY_MAP.get(_name, set()),
        })

# Public list maintained for backwards compatibility
FUNCTIONAL_GROUP_PATTERNS: list[tuple[str, Chem.Mol]] = [
    (p["name"], p["pattern"]) for p in _FUNCTIONAL_GROUP_PATTERNS_INTERNAL
]

# Maintain for compatibility
FUNCTIONAL_GROUP_SMARTS = dict(zip(_df.trivialname, zip(_df.SMARTS, _df.pattern_type, strict=False), strict=False))


def get_functional_group_matches(mol: Chem.Mol, include_overshadowed: bool = False) -> pd.DataFrame:
    """
    Applies functional group patterns and returns a DataFrame with detailed metadata.

    Args:
        mol: The RDKit molecule to analyze.
        include_overshadowed: If False, filters out groups that are part of a larger
            detected group.

    Returns:
        A pandas DataFrame where each row is a functional group match.
        Columns:
            - name: trivial name of the functional group.
            - atom_ids: tuple of atom indices in the match.
            - smarts: the SMARTS or SMILES string used for matching.
            - group: the group category (functional_group, biological, cyclic).
            - pattern_type: either 'smarts' or 'smiles'.


    WARNING: These SMARTS patterns are taken from
    https://github.com/torbengutermuth/SmartChemist/blob/main/smarts/License_for_patterns_here
    These patterns have aCreative Commons Attribution-NoDerivatives 4.0 International Public License
    If you use these patterns in your own code, remember to credit them!
    Do NOT modify the patterns without permission!
    """
    all_raw_matches = []
    for idx, p_info in enumerate(_FUNCTIONAL_GROUP_PATTERNS_INTERNAL):
        matches = mol.GetSubstructMatches(p_info["pattern"])
        for atom_ids in matches:
            all_raw_matches.append({
                "name": p_info["name"],
                "atom_ids": tuple(sorted(atom_ids)),
                "atom_set": set(atom_ids),
                "priority": idx,
                "smarts": p_info["smarts"],
                "group": p_info["group"],
                "pattern_type": p_info["pattern_type"],
            })

    if not include_overshadowed:
        filtered_matches = []
        for i, m1 in enumerate(all_raw_matches):
            is_overshadowed = False
            for j, m2 in enumerate(all_raw_matches):
                if i == j:
                    continue

                # Rule 1: Structural inclusion (strictly a subset)
                if m1["atom_set"] < m2["atom_set"]:
                    is_overshadowed = True
                    break

                # Rule 2: Identical atom sets
                if m1["atom_set"] == m2["atom_set"] and m1["name"] in HIERARCHY_MAP.get(m2["name"], set()):
                    is_overshadowed = True
                    break

            if not is_overshadowed:
                filtered_matches.append(m1)
        results = filtered_matches
    else:
        results = all_raw_matches

    rows = []
    for m in results:
        rows.append({
            "name": m["name"],
            "atom_ids": m["atom_ids"],
            "smarts": m["smarts"],
            "group": m["group"],
            "pattern_type": m["pattern_type"],
        })

    return pd.DataFrame(rows)


def get_all_functional_group_patterns() -> pd.DataFrame:
    """
    Returns a DataFrame containing all defined functional group patterns and their metadata.

    Returns:
        A pandas DataFrame where each row is a functional group pattern.
        Columns:
            - name: trivial name of the functional group.
            - smarts: the SMARTS or SMILES string.
            - group: the group category (functional_group, biological, cyclic).
            - pattern_type: either 'smarts' or 'smiles'.

    WARNING: These SMARTS patterns are taken from
    https://github.com/torbengutermuth/SmartChemist/blob/main/smarts/License_for_patterns_here
    These patterns have aCreative Commons Attribution-NoDerivatives 4.0 International Public License
    If you use these patterns in your own code, remember to credit them!
    Do NOT modify the patterns without permission!
    """
    rows = []
    for p_info in _FUNCTIONAL_GROUP_PATTERNS_INTERNAL:
        rows.append({
            "name": p_info["name"],
            "smarts": p_info["smarts"],
            "group": p_info["group"],
            "pattern_type": p_info["pattern_type"],
            "pattern": p_info["pattern"],
            "priority": p_info["priority"],
            "hierarchy": p_info["hierarchy"],
        })
    return pd.DataFrame(rows)
