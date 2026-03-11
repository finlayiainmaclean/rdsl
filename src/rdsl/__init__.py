import sys

from loguru import logger

from rdsl.functional_groups import (
    get_all_functional_group_patterns,
    get_functional_group_matches,
)
from rdsl.highlight import highlight_atoms
from rdsl.select import SelectionResult, select_atom_ids, select_atoms, select_molecule
from rdsl.widgets import display_functional_groups

# Configure loguru to use INFO level by default
logger.remove()
logger.add(sys.stderr, level="INFO")


__all__ = [
    "SelectionResult",
    "display_functional_groups",
    "get_all_functional_group_patterns",
    "get_functional_group_matches",
    "highlight_atoms",
    "select_atom_ids",
    "select_atoms",
    "select_molecule",
]
