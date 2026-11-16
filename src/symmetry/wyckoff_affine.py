from fractions import Fraction
import re
import numpy as np

def _parse_number(term: str) -> float:
    """Parse a numeric token like '2', '-3', '1/4', '-2/3'."""
    term = term.strip()
    return float(Fraction(term))


def _split_signed_terms(expr: str) -> list[str]:
    """
    Split expression into signed terms.
    Example:
        'x-y+1/4' -> ['x', '-y', '+1/4']
        '-2x+z'   -> ['-2x', '+z']
    """
    expr = expr.replace(" ", "")
    if not expr:
        raise ValueError("Empty expression")

    if expr[0] not in "+-":
        expr = "+" + expr

    terms = re.findall(r"[+-][^+-]+", expr)
    return terms


def _parse_term(term: str):
    """
    Parse one signed term into either:
      - ('var', variable_name, coefficient)
      - ('const', value)

    Supported examples:
      +x, -x, +2x, -3/2x, +z, +1/4, -2/3
    """
    sign = -1.0 if term[0] == "-" else 1.0
    body = term[1:]

    # variable term: optional numeric coeff followed by x/y/z
    m = re.fullmatch(r"(?:(\d+(?:/\d+)?)|(\d*\.\d+))?([xyz])", body)
    if m:
        int_or_frac, decimal, var = m.groups()

        if int_or_frac is not None:
            coeff = _parse_number(int_or_frac)
        elif decimal is not None:
            coeff = float(decimal)
        else:
            coeff = 1.0

        return ("var", var, sign * coeff)

    # constant term
    try:
        value = sign * _parse_number(body)
        return ("const", value)
    except Exception as e:
        raise ValueError(f"Could not parse term '{term}'") from e


def wp_representative_to_A_b(wp_representative_str: str):
    """
    Convert a Wyckoff representative string like:
        '1/3, 2/3, z'
        'x,2x,1/4'
        'x, x-y, z'
        'x+1/2, -2x, z'
    into:
        A: (3, 3) numpy array
        b: (3,) numpy array
        free_vars: list[str]   # variable order used for columns of A

    Convention:
      - Columns of A correspond to free variables in order of first appearance.
      - Unused columns are zero-padded.
      - At most 3 free variables are allowed.
    """
    coords = [c.strip() for c in wp_representative_str.split(",")]
    if len(coords) != 3:
        raise ValueError(
            f"Expected 3 coordinate expressions separated by commas, got {len(coords)}: {wp_representative_str}"
        )

    # First pass: determine variable order by first appearance
    free_vars = []
    for expr in coords:
        terms = _split_signed_terms(expr)
        for term in terms:
            parsed = _parse_term(term)
            if parsed[0] == "var":
                var = parsed[1]
                if var not in free_vars:
                    free_vars.append(var)

    if len(free_vars) > 3:
        raise ValueError(f"Too many free variables ({len(free_vars)}), expected at most 3.")

    var_to_col = {var: i for i, var in enumerate(free_vars)}

    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    # Second pass: fill A and b
    for row, expr in enumerate(coords):
        terms = _split_signed_terms(expr)
        for term in terms:
            parsed = _parse_term(term)

            if parsed[0] == "const":
                b[row] += parsed[1]

            elif parsed[0] == "var":
                _, var, coeff = parsed
                col = var_to_col[var]
                A[row, col] += coeff

            else:
                raise RuntimeError(f"Unexpected parsed term: {parsed}")
    
    #free_mask = np.zeros(3, dtype=float)
    #free_mask[:len(free_vars)] = 1.0

    return A, b #, free_mask