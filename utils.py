import itertools as itt

def kronecker(a, b):
  return 1 if a==b else 0

def compute_delta_terms(indices):
    """
    Calcule les symboles de Kronecker pour une liste d'indices.
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
    Génère une liste de tuples représentant les indices formés en combinant les éléments de 'coords', répétés 'n' fois.

    Args:
        coords (list or tuple): Liste des labels de coordonnées (par exemple, ('x', 'y', 'z')).
        n (int): Nombre de fois que les coordonnées sont répétées.
        permute (bool): Si True, génère toutes les permutations possibles (avec répétition).
                        Si False, génère les combinaisons avec répétition.

    Returns:
        list: Liste de tuples représentant les indices.
    """
  if permute:
        iterator = itt.product(coords, repeat=n)
  else:
        iterator = itt.combinations_with_replacement(coords, n)
  
  if as_string:
        return [''.join(item) for item in iterator]
  else:
        return list(iterator)


def circular_permutation(sequence):
    """
    Génère toutes les rotations circulaires de la séquence donnée.

    Args:
        sequence (list): La séquence à faire tourner.

    Returns:
        list: Liste de toutes les rotations circulaires de la séquence.
    """
    n = len(sequence)
    return [sequence[i:] + sequence[:i] for i in range(n)]
