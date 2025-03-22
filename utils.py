import itertools as itt
import sympy as sp
import math
from collections import Counter

def kronecker(a, b):
  return 1 if a==b else 0

def compute_delta_terms(indices):
    """
    Compute the Kronecker symbols for a list of indices.
    """
    delta_terms = {}
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            key = f'{indices[i]}{indices[j]}'
            delta_terms[key] = kronecker(indices[i], indices[j])
    return delta_terms


def compute_kronecker_third_order_coeff(indices, var):
    # Get indices
    a,b,c = indices
    
    # Compute the Kronecker delta terms
    delta_ab = kronecker(a, b)
    delta_ac = kronecker(a, c)
    delta_bc = kronecker(b, c)
    
    # Compute the third order term
    return delta_ab*var[c]+delta_ac*var[b]+delta_bc*var[a]

def compute_kronecker_fourth_order(indices):
    # Get indices
    a,b,c, d = indices

    # Compute the Kronecker delta terms
    delta_ab = kronecker(a, b)
    delta_ac = kronecker(a, c)
    delta_ad = kronecker(a, d)
    delta_bc = kronecker(b, c)
    delta_bd = kronecker(b, d)
    delta_cd = kronecker(c, d)

    # Compute the coefficients for the fourth order terms
    delta_delta_abcd = delta_ab*delta_cd+delta_ac*delta_bd+delta_ad*delta_bc
    return delta_delta_abcd

def compute_kronecker_fourth_order_coeff(indices, var):
    # Get indices
    a,b,c, d = indices

    # Compute the Kronecker delta terms
    delta_ab = kronecker(a, b)
    delta_ac = kronecker(a, c)
    delta_ad = kronecker(a, d)
    delta_bc = kronecker(b, c)
    delta_bd = kronecker(b, d)
    delta_cd = kronecker(c, d)

    # Compute the coefficients for the fourth order terms
    delta_delta_abcd = delta_ab*delta_cd+delta_ac*delta_bd+delta_ad*delta_bc
    dvar_abcd_1 = delta_ab*var[a+b] + delta_ac * var[a+c] + delta_ad*var[a+d]
    dvar_abcd_2 = delta_bc*var[b+c] + delta_bd * var[b+d] + delta_cd*var[c+d]
    dvar_abcd = dvar_abcd_1 + dvar_abcd_2
    return delta_delta_abcd, dvar_abcd
    
    # Compute the Kronecker delta terms and the corresponding product with variables
    delta_ab = kronecker(a, b) * var.get(c, 0)
    delta_ac = kronecker(a, c) * var.get(b, 0)
    delta_bc = kronecker(b, c) * var.get(a, 0)
    return delta_ab*var[c]+delta_ac*var[b]+delta_bc*var[a]
    

def generate_indices(coords, n, permute=True, as_string=False):
    """
    Generates a list of tuples representing the indices formed by combining the elements of 'coords', repeated 'n' times.

    Args:
        coords (list or tuple): List of coordinate labels (e.g., ('x', 'y', 'z')).
        n (int): Number of times the coordinates are repeated.
        permute (bool): If True, generates all possible permutations (with repetition).
                        If False, generates combinations with repetition.

    Returns:
        list: List of tuples representing the indices.
    """
    if permute:
            iterator = itt.product(coords, repeat=n)
    else:
            iterator = itt.combinations_with_replacement(coords, n)
    
    if as_string:
            return [''.join(item) for item in iterator]
    else:
            return list(iterator)
  

def compute_multiplicity(order, indices):
    """
    Compute the multiplicity of the indices based on the given order.

    The multiplicity is calculated as the factorial of the order divided by
    the product of factorials of the counts of each index. This ensures that
    each index is counted the correct number of times in the sum.

    Parameters:
    - order (int): The order of the polynomial.
    - indices (list): A list of indices.

    Returns:
    - multiplicity (int): The multiplicity of the indices.

    Example:
    >>> compute_multiplicty(2, ['x', 'x', 'y'])
    1
    >>> compute_multiplicty(2, ['x', 'y', 'y'])
    2
    """
    multiplicity = math.factorial(order)
    counts = Counter(indices)
    for count in counts.values():
        multiplicity //= math.factorial(count)
    return multiplicity


def generate_circular_permutation(sequence):
    """
    Generate all circular rotations of the given sequence.

    Args:
        sequence (list): The sequence to rotate.

    Returns:
        list: List of all circular rotations of the sequence.
    """
    n = len(sequence)
    return [sequence[i:] + sequence[:i] for i in range(n)]

# Function to divide each element of two matrices
def matrix_divide_elementwise(a, b):
    """
    Divide each element of matrix `a` by the corresponding element of matrix `b`.

    Parameters:
    a (sympy.Matrix): The matrix to be divided.
    b (sympy.Matrix): The matrix used for division.

    Returns:
    sympy.Matrix: A new matrix containing the element-wise division results.

    Raises:
    ValueError: If the dimensions of `a` and `b` are not the same.

    Notes:
    - If an element in `b` is zero, the corresponding element in the result matrix will be NaN.
    """

    if a.shape != b.shape:
        raise ValueError("The matrices must have the same dimension.")
    
    # Create a new matrix to store the results
    result = sp.Matrix(a.shape[0], a.shape[1], lambda i, j: 0)
    
    # Loop over the elements of the matrices to perform the division
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] == 0:
                result[i, j] = sp.nan  # Return NaN if b[i] is zero
            else:
                result[i, j] = a[i, j] / b[i, j]
    return result


def print_mat(matrix):
    return [h for slist in matrix.tolist() for h in slist]
