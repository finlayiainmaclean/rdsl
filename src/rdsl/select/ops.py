from typing import ClassVar

import numpy as np
import pandas as pd
from rdkit import Chem

from rdsl.functional_groups import get_functional_group_matches
from rdsl.select.base import (
    _ALIASES,
    _FEATURE_FACTORY,
    _FEATURE_MAP,
    PREDEFINED_SMARTS,
    BaseOp,
    ConformerError,
    InvalidPatternError,
    SelectionParserError,
)


class CompareOp(BaseOp):
    """Numeric comparison: formal_charge < 1, partial_charge > -0.5, mass >= 12.0"""

    _OPS: ClassVar[dict] = {
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, tokens):
        kw, op, val = tokens
        self.kw = _ALIASES[kw]
        self.op = op
        self.val = val

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        col = ctx[self.kw]
        fn = self._OPS[self.op]
        # Handle None/NaN gracefully
        return np.array(
            [fn(v, self.val) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else False for v in col],
            dtype=bool,
        )

    def __repr__(self):
        return f"compare(kw={self.kw}, op='{self.op}', val={self.val})"


class FlagOp(BaseOp):
    def __init__(self, tokens):
        kw = _ALIASES[tokens[0]]
        self.kw = kw
        self.is_smarts = kw in PREDEFINED_SMARTS

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        if self.kw == "all":
            return np.ones(len(ctx), dtype=bool)
        if self.kw == "none":
            return np.zeros(len(ctx), dtype=bool)

        # Lazy feature evaluation
        if self.kw in _FEATURE_MAP and self.kw not in ctx.columns:
            self._evaluate_feature(mol, ctx)

        # Lazy SMARTS evaluation
        if self.is_smarts and self.kw not in ctx.columns:
            self._evaluate_smarts(mol, ctx)

        return ctx[self.kw].values.flatten()

    def _evaluate_feature(self, mol: Chem.Mol, ctx: pd.DataFrame) -> None:
        family = _FEATURE_MAP[self.kw]
        features = _FEATURE_FACTORY.GetFeaturesForMol(mol)
        mask = np.zeros(len(ctx), dtype=bool)
        for feat in features:
            if feat.GetFamily() == family:
                mask[list(feat.GetAtomIds())] = True
        ctx[self.kw] = mask

    def _evaluate_smarts(self, mol: Chem.Mol, ctx: pd.DataFrame) -> None:
        smi_or_smarts, pattern_type = PREDEFINED_SMARTS[self.kw]
        if pattern_type == "smarts":
            patterns = [Chem.MolFromSmarts(smi_or_smarts)]
        elif pattern_type == "smiles":
            patterns = [Chem.MolFromSmiles(smi_or_smarts)]
        elif pattern_type == "smarts_list":
            patterns = [Chem.MolFromSmarts(s) for s in smi_or_smarts]
        else:
            raise InvalidPatternError(kw=self.kw, pattern_type=pattern_type)

        patterns = [p for p in patterns if p is not None]

        matched_atoms = set()
        for pattern in patterns:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                matched_atoms.update(match)

        mask = np.zeros(len(ctx), dtype=bool)
        mask[list(matched_atoms)] = True
        ctx[self.kw] = mask

    def __repr__(self):
        return f"flag(kw={self.kw})"


class AttrOp(BaseOp):
    def __init__(self, tokens):
        kw, *args = tokens
        self.kw = _ALIASES[kw]
        self.args = {*args}

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        return ctx[self.kw].isin(self.args).values

    def __repr__(self):
        return f"attr(kw={self.kw}, args={self.args})"


class SmartsOp(BaseOp):
    """SMARTS pattern matching."""

    def __init__(self, tokens):
        self.pattern = tokens[1]

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        pattern = Chem.MolFromSmarts(self.pattern)
        if pattern is None:
            raise InvalidPatternError(pattern=self.pattern)

        matches = mol.GetSubstructMatches(pattern)
        matched_atoms = set()
        for match in matches:
            matched_atoms.update(match)

        mask = np.zeros(len(ctx), dtype=bool)
        mask[list(matched_atoms)] = True
        return mask

    def __repr__(self):
        return f"smarts(pattern={self.pattern})"


class FunctionalOp(BaseOp):
    """Functional group matching by name: functional 'alcohol' or functional ketone"""

    def __init__(self, tokens):
        # tokens[0] is the keyword, tokens[1:] are the names
        self.names = {t.lower() for t in tokens[1:]}

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        matches_df = get_functional_group_matches(mol, include_overshadowed=True)
        mask = np.zeros(len(ctx), dtype=bool)

        if matches_df.empty:
            return mask

        # Filter by names (trivial name in CSV is already lowercased)
        matched_groups = matches_df[matches_df["name"].isin(self.names)]

        for _, row in matched_groups.iterrows():
            mask[list(row["atom_ids"])] = True

        return mask

    def __repr__(self):
        return f"functional(names={self.names})"


class RingSizeOp(BaseOp):
    """Select atoms in rings of specific size."""

    def __init__(self, tokens):
        kw, *sizes = tokens
        self.kw = _ALIASES[kw]
        self.sizes = {*sizes}

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        ri = mol.GetRingInfo()
        mask = np.zeros(len(ctx), dtype=bool)
        for size in self.sizes:
            for ring in ri.AtomRings():
                if len(ring) == size:
                    mask[list(ring)] = True
        return mask

    def __repr__(self):
        return f"ringsize(sizes={self.sizes})"


class UnaryOp(BaseOp):
    def __init__(self, tokens):
        op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        if self.op.lower() == "not":
            return ~self.rhs.apply(mol, ctx)
        raise NotImplementedError(f"unsupported unary operation: {self.op}")

    def __repr__(self):
        return f"unary(op='{self.op}', rhs={self.rhs})"


class BinaryOp(BaseOp):
    def __init__(self, tokens):
        ops = {*tokens[0][1::2]}
        if len(ops) != 1:
            raise SelectionParserError(expected="1 operator", got=len(ops))
        self.op = _ALIASES[ops.pop()]
        self.matchers = tokens[0][::2]

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        matches = [m.apply(mol, ctx) for m in self.matchers]
        lhs = matches[0]
        for rhs in matches[1:]:
            if self.op.lower() in ("and", "in"):
                lhs = lhs & rhs
            elif self.op.lower() == "or":
                lhs = lhs | rhs
            else:
                raise NotImplementedError(f"unsupported binary operation: {self.op}")
        return lhs

    def __repr__(self):
        return f"binary(op='{self.op}', matchers={self.matchers})"


class DistOp(BaseOp):
    """Distance-based selection (Angstroms, requires 3D conformer)."""

    def __init__(self, tokens):
        self.lhs, op, self.dist, of_kw, self.rhs = tokens[0]
        self.op = _ALIASES[op]
        if of_kw != "of":
            raise SelectionParserError(expected="'of'", got=of_kw)

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        if mol.GetNumConformers() == 0:
            raise ConformerError()

        lhs_mask = self.lhs.apply(mol, ctx)
        rhs_mask = self.rhs.apply(mol, ctx)
        lhs_indices = np.where(lhs_mask)[0]
        rhs_indices = np.where(rhs_mask)[0]

        conf = mol.GetConformer()
        result = np.zeros(len(ctx), dtype=bool)

        for i in lhs_indices:
            pos_i = conf.GetAtomPosition(int(i))
            for j in rhs_indices:
                if i == j:
                    continue
                pos_j = conf.GetAtomPosition(int(j))
                dist = pos_i.Distance(pos_j)
                if (self.op == "within" and dist <= self.dist) or (self.op == "beyond" and dist > self.dist):
                    result[i] = True
                    break
        return result

    def __repr__(self):
        return f"dist(lhs={self.lhs}, op='{self.op}', dist={self.dist}, rhs={self.rhs})"


class ExtendOp(BaseOp):
    """Expand selection by N bonds."""

    def __init__(self, tokens):
        if len(tokens[0]) == 2:  # prefix mode
            op_kw = tokens[0][0]
            op_name = _ALIASES[op_kw]
            self.rhs = tokens[0][1]
            self.n = 1
            self.exclude_base = op_name == "neighbor"
        else:  # postfix extend
            self.rhs = tokens[0][0]
            self.n = int(tokens[0][2])
            self.exclude_base = False

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        base_mask = self.rhs.apply(mol, ctx)
        base_indices = set(np.where(base_mask)[0])
        dist_matrix = Chem.GetDistanceMatrix(mol)
        result = np.zeros(len(ctx), dtype=bool)
        for i in range(len(ctx)):
            if not base_indices:
                continue
            min_dist_to_base = min(dist_matrix[i, j] for j in base_indices)
            if self.exclude_base:
                if 0 < min_dist_to_base <= self.n:
                    result[i] = True
            else:
                if min_dist_to_base <= self.n:
                    result[i] = True
        return result

    def __repr__(self):
        return f"extend(rhs={self.rhs}, n={self.n}, exclude_base={self.exclude_base})"


class AroundOp(BaseOp):
    """Postfix distance selection: selection around 12.3"""

    def __init__(self, tokens):
        self.rhs = tokens[0][0]
        self.dist = float(tokens[0][2])

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        if mol.GetNumConformers() == 0:
            raise ConformerError()
        rhs_mask = self.rhs.apply(mol, ctx)
        rhs_indices = np.where(rhs_mask)[0]
        if len(rhs_indices) == 0:
            return np.zeros(len(ctx), dtype=bool)
        conf = mol.GetConformer()
        result = np.zeros(len(ctx), dtype=bool)
        rhs_positions = [conf.GetAtomPosition(int(j)) for j in rhs_indices]
        for i in range(len(ctx)):
            pos_i = conf.GetAtomPosition(i)
            # Find if this atom is within dist of ANY atom in rhs
            for pos_j in rhs_positions:
                if pos_i.Distance(pos_j) <= self.dist:
                    result[i] = True
                    break
        return result

    def __repr__(self):
        return f"around(rhs={self.rhs}, dist={self.dist})"


class GapOp(BaseOp):
    """Postfix gap selection: selection gap 1.2"""

    def __init__(self, tokens):
        self.rhs = tokens[0][0]
        self.threshold = float(tokens[0][2])

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        if mol.GetNumConformers() == 0:
            raise ConformerError()
        rhs_mask = self.rhs.apply(mol, ctx)
        rhs_indices = np.where(rhs_mask)[0]
        if len(rhs_indices) == 0:
            return np.zeros(len(ctx), dtype=bool)
        conf = mol.GetConformer()
        result = np.zeros(len(ctx), dtype=bool)
        pt = Chem.GetPeriodicTable()
        atoms = mol.GetAtoms()
        vdw_radii = [pt.GetRvdw(a.GetAtomicNum()) for a in atoms]
        rhs_data = [(conf.GetAtomPosition(int(j)), vdw_radii[int(j)]) for j in rhs_indices]
        for i in range(len(ctx)):
            pos_i = conf.GetAtomPosition(i)
            r_i = vdw_radii[i]
            is_separated = True
            for pos_j, r_j in rhs_data:
                gap = pos_i.Distance(pos_j) - r_i - r_j
                if gap < self.threshold:
                    is_separated = False
                    break
            if is_separated:
                result[i] = True
        return result

    def __repr__(self):
        return f"gap(rhs={self.rhs}, threshold={self.threshold})"


class ExpandOp(BaseOp):
    def __init__(self, tokens):
        if len(tokens) != 1:
            raise SelectionParserError(expected="1 token", got=len(tokens))
        op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        rhs = self.rhs.apply(mol, ctx)
        if self.op.lower() in {"bymolecule", "bymol", "bm"}:
            frags = Chem.GetMolFrags(mol, asMols=False)
            selected_frags = set()
            for frag_idx, frag_atoms in enumerate(frags):
                if any(rhs[atom_idx] for atom_idx in frag_atoms):
                    selected_frags.add(frag_idx)
            result = np.zeros(len(ctx), dtype=bool)
            for frag_idx in selected_frags:
                for atom_idx in frags[frag_idx]:
                    result[atom_idx] = True
            return result
        if self.op.lower() in {"bychain", "bc"}:
            selected_chains = set(ctx.loc[rhs, "chain"].dropna().unique())
            return ctx["chain"].isin(selected_chains).values
        if self.op.lower() in {"byres", "br"}:
            selected_residues = set(ctx.loc[rhs, "residue"].dropna().unique())
            return ctx["residue"].isin(selected_residues).values
        if self.op.lower() in {"byring", "bring", "byr"}:
            ri = mol.GetRingInfo()
            atom_rings = ri.AtomRings()
            selected_indices = np.where(rhs)[0]
            if len(selected_indices) == 0:
                return np.zeros(len(ctx), dtype=bool)

            connected_rings = set()
            for ring_idx, ring in enumerate(atom_rings):
                ring_set = set(ring)
                for idx in selected_indices:
                    # Case 1: Atom is in the ring
                    if idx in ring_set:
                        connected_rings.add(ring_idx)
                        break
                    # Case 2: Atom is bonded to an atom in the ring
                    atom = mol.GetAtomWithIdx(int(idx))
                    if any(neighbor.GetIdx() in ring_set for neighbor in atom.GetNeighbors()):
                        connected_rings.add(ring_idx)
                        break

            result = np.zeros(len(ctx), dtype=bool)
            for ring_idx in connected_rings:
                for atom_idx in atom_rings[ring_idx]:
                    result[atom_idx] = True
            return result
        if self.op.lower() in {"byfunctional", "byfg", "bfg"}:
            matches_df = get_functional_group_matches(mol, include_overshadowed=True)
            if matches_df.empty:
                return np.zeros(len(ctx), dtype=bool)

            selected_indices = np.where(rhs)[0]
            if len(selected_indices) == 0:
                return np.zeros(len(ctx), dtype=bool)

            # Map each atom to the list of groups (as atom_ids tuples) it belongs to
            atom_to_groups = {}
            for _, row in matches_df.iterrows():
                group_atoms = row["atom_ids"]
                for atom_idx in group_atoms:
                    if atom_idx not in atom_to_groups:
                        atom_to_groups[atom_idx] = []
                    atom_to_groups[atom_idx].append(group_atoms)

            result_indices = set()
            for idx in selected_indices:
                groups = atom_to_groups.get(idx)
                if not groups:
                    continue

                # Find the largest size among groups containing this specific atom
                max_size = max(len(g) for g in groups)

                # Add all atoms from all groups that have this maximum size for THIS atom
                for g in groups:
                    if len(g) == max_size:
                        result_indices.update(g)

            result = np.zeros(len(ctx), dtype=bool)
            for idx in result_indices:
                result[idx] = True
            return result
        raise NotImplementedError(f"unsupported expansion operation: {self.op}")

    def __repr__(self):
        return f"expand(op='{self.op}', rhs={self.rhs})"


class FirstLastOp(BaseOp):
    def __init__(self, tokens):
        op, self.rhs = tokens[0]
        self.op = _ALIASES[op]

    def apply(self, mol: Chem.Mol, ctx: pd.DataFrame) -> np.ndarray:
        mask = self.rhs.apply(mol, ctx)
        indices = np.where(mask)[0]
        result = np.zeros(len(ctx), dtype=bool)
        if len(indices) > 0:
            idx = int(np.min(indices) if self.op == "first" else np.max(indices))
            result[idx] = True
        return result

    def __repr__(self):
        return f"firstlast(op='{self.op}', rhs={self.rhs})"
