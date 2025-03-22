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

# Constants for the isotropic fourth order terms
ISO_4_ONE_9_3P2SQRT3 = 4 * ONE_9 * (3 + 2 * SQRT3)
ISO_4_ONE_9_3MSQRT3 = 4 * ONE_9 * (3 - SQRT3)

# Standard lattice for indices
COORDS = ('x', 'y', 'z')

# Computation of the factor 1/[n!*cs^(2n)] for orders 0, 1, 2, 3, 4
def compute_one_on_factorial_cs2_n(order, cs2):
    """
    Calculate a list of factors 1 / [n! * cs2^(n)] for n ranging from 0 to 'order' inclusive.

    Args:
        order (int): The maximum order for which to calculate the factor.
        cs2 (sympy.Expr or float): The value of cs^2 used in the calculation.

    Returns:
        list: List of calculated factors for n ranging from 0 to 'order'.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError("The 'order' parameter must be a non-negative integer.")
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
