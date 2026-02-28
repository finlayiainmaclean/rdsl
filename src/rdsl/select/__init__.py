from rdsl.select.core import SelectionResult, select_atom_ids, select_atoms, select_molecule
from rdsl.select.utils import _create_context

select = select_atom_ids

__all__ = ["SelectionResult", "_create_context", "select", "select_atoms", "select_molecule"]
