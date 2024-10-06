import sympy as sp

MAX_ORDER = 4

# Rationals
ONE_2 = sp.Rational(1, 2)
ONE_3 = sp.Rational(1, 3)
ONE_4 = sp.Rational(1, 4)
ONE_6 = sp.Rational(1, 6)
ONE_9 = sp.Rational(1, 9)
ONE_12 = sp.Rational(1, 12)
ONE_18 = sp.Rational(1, 18)
ONE_27 = sp.Rational(1, 27)
ONE_36 = sp.Rational(1, 36)
ONE_54 = sp.Rational(1, 54)
ONE_216 = sp.Rational(1, 216)

# SQUARE ROOTS
SQRT2 = sp.sqrt(2)
SQRT3 = sp.sqrt(3)

# Constants for the istropic fourth order terms
ISO_4_ONE_9_3P2SQRT3 = 4 * ONE_9 * (3 + 2 * SQRT3)
ISO_4_ONE_9_3MSQRT3 = 4 * ONE_9 * (3 - SQRT3)

# Standard lattice for indices
COORDS = ('x', 'y', 'z')

# Computation of the factor 1/[n!*cs^(2n)] for orders 0, 1, 2, 3, 4
def compute_one_on_factorial_cs2_n(order, cs2):
    """
    Calcule une liste des facteurs 1 / [n! * cs2^(n)] pour n variant de 0 à 'order' inclus.

    Args:
        order (int): L'ordre maximal pour lequel calculer le facteur.
        cs2 (sympy.Expr or float): La valeur de cs^2 utilisée dans le calcul.

    Returns:
        list: Liste des facteurs calculés pour n allant de 0 à 'order'.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError("Le paramètre 'order' doit être un entier non négatif.")
    return [sp.Rational(1, sp.factorial(o)*cs2**o) for o in range(order+1)]


# Constants used in equilibrium function


# Constant matrices depending on the lattice
def get_constant_mat0(Q):
    mat0 = sp.Matrix([0 for _ in range(Q)])
    return mat0

def get_constant_mat1(Q):
    mat1 = sp.Matrix([1 for _ in range(Q)])
    return mat1

def get_constant_d0i(Q):
    d0i = sp.Matrix([0 for _ in range(Q)])
    d0i[0] = 1
    return d0i


