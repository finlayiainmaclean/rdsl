import functools

from pyparsing import (
    CaselessKeyword,
    MatchFirst,
    OpAssoc,
    ParseResults,
    QuotedString,
    Regex,
    infix_notation,
)

from rdsl.select.base import (
    _AROUND_OPS,
    _ATTR_OPS,
    _BINARY_OPS,
    _BOND_OPS,
    _DECIMAL,
    _DIST_OPS,
    _EXPANSION_OPS,
    _FIRSTLAST_OPS,
    _FLAG_OPS,
    _GAP_OPS,
    _IDENTIFIER,
    _INTEGER,
    _NEIGHBOR_OPS,
    _RING_OPS,
    _SMARTS_OP,
    _FUNCTIONAL_OP,
    _SUBSET_OPS,
    _UNARY_OPS,
    _flatten,
)
from rdsl.select.ops import (
    AroundOp,
    AttrOp,
    BinaryOp,
    CompareOp,
    DistOp,
    ExpandOp,
    ExtendOp,
    FirstLastOp,
    FlagOp,
    GapOp,
    RingSizeOp,
    SmartsOp,
    FunctionalOp,
    UnaryOp,
)


def _create_parser():
    """Create the parser for selection expressions."""
    flag_ops = [CaselessKeyword(kw).set_parse_action(FlagOp) for op in _FLAG_OPS for kw in _flatten(op)]

    # Comparison ops: formal_charge < 1, partial_charge >= -0.5, mass > 12.0
    _NUMERIC_ATTR_OPS = [op for op in _ATTR_OPS if op[2] in (_DECIMAL, _INTEGER)]
    _COMPARE_OP = Regex(r"<=|>=|!=|==|<|>")
    compare_ops = [
        (CaselessKeyword(kw) + _COMPARE_OP + _DECIMAL).set_parse_action(CompareOp)
        for op in _NUMERIC_ATTR_OPS
        for kw in _flatten(op)
    ]

    attr_ops = [(CaselessKeyword(kw) + op[2]).set_parse_action(AttrOp) for op in _ATTR_OPS for kw in _flatten(op)]

    # SMARTS pattern: smarts "pattern" or smarts 'pattern'
    smarts_pattern = QuotedString('"', unquote_results=True) | QuotedString("'", unquote_results=True)
    smarts_ops = [(CaselessKeyword(kw) + smarts_pattern).set_parse_action(SmartsOp) for op in _SMARTS_OP for kw in _flatten(op)]
    
    # Functional group pattern: functional "alcohol", functional 'alcohol', or functional ketone
    functional_ops = [(CaselessKeyword(kw) + _IDENTIFIER).set_parse_action(FunctionalOp) for op in _FUNCTIONAL_OP for kw in _flatten(op)]

    ring_ops = [(CaselessKeyword(kw) + _INTEGER).set_parse_action(RingSizeOp) for op in _RING_OPS for kw in _flatten(op)]

    # compare_ops must come before attr_ops in MatchFirst to avoid partial consumption
    selector_op = MatchFirst(op for op in flag_ops + compare_ops + attr_ops + smarts_ops + functional_ops + ring_ops)

    unary_ops = [(CaselessKeyword(kw), 1, OpAssoc.RIGHT, UnaryOp) for op in _UNARY_OPS for kw in _flatten(op)]
    binary_ops = [(CaselessKeyword(kw), 2, OpAssoc.LEFT, BinaryOp) for op in _BINARY_OPS for kw in _flatten(op)]
    subset_ops = [(CaselessKeyword(kw), 2, OpAssoc.LEFT, BinaryOp) for op in _SUBSET_OPS for kw in _flatten(op)]

    expand_ops = [(CaselessKeyword(kw), 1, OpAssoc.RIGHT, ExpandOp) for op in _EXPANSION_OPS for kw in _flatten(op)]
    firstlast_ops = [(CaselessKeyword(kw), 1, OpAssoc.RIGHT, FirstLastOp) for op in _FIRSTLAST_OPS for kw in _flatten(op)]

    _EXTEND_N = Regex(r"\d+").set_parse_action(lambda t: int(t[0]))
    extend_ops = [
        (CaselessKeyword("extend") + _EXTEND_N, 1, OpAssoc.LEFT, ExtendOp),
    ]
    extend_ops += [
        (CaselessKeyword(kw), 1, OpAssoc.RIGHT, ExtendOp) for op in (_BOND_OPS + _NEIGHBOR_OPS) for kw in _flatten(op)
    ]

    around_ops = [(CaselessKeyword(kw) + _DECIMAL, 1, OpAssoc.LEFT, AroundOp) for op in _AROUND_OPS for kw in _flatten(op)]

    gap_ops = [(CaselessKeyword(kw) + _DECIMAL, 1, OpAssoc.LEFT, GapOp) for op in _GAP_OPS for kw in _flatten(op)]

    dist_ops = [
        (CaselessKeyword(kw) + _DECIMAL + CaselessKeyword("of"), 2, OpAssoc.LEFT, DistOp) for op in _DIST_OPS for kw in _flatten(op)
    ]

    expr = infix_notation(
        selector_op,
        unary_ops + binary_ops + expand_ops + firstlast_ops + extend_ops + around_ops + gap_ops + dist_ops + subset_ops,
    )
    return expr


# The expression parser
_PARSER = _create_parser()


@functools.lru_cache
def _parse_expr(expr: str) -> ParseResults:
    """Parse a selection expression."""
    return _PARSER.parse_string(expr, parse_all=True)
