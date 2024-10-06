import sympy as sp
import functools
import numpy as np

from constants import *
from utils import *


class HermitePolynomials:
    def __init__(self, Ci, cs2, order, Q, D):
        """
        Initialise les polynômes d'Hermite.

        Args:
            Ci (dict): Dictionnaire des vecteurs de vitesse.
            cs2 (sympy.Expr): Vitesse du son au carré.
            order (int): Ordre maximal des polynômes.
            Q (int): Nombre de directions du réseau.
            D (int): Dimension spatiale.
        """
        self.Ci = Ci
        self.order = order
        self.Q = Q
        self.D = D
        self.coords = COORDS[:self.D]
        self.Hi = {}
        self.cs2 = cs2
        self.cs4 = cs2 ** 2
        self.cs8 = self.cs4*self.cs4

        self._compute_hermite_polynomials()

    def _compute_hermite_polynomials(self):
        """
        Génère les polynômes d'Hermite non dimensionnels jusqu'à l'ordre spécifié.
        Génère le dictionnaire Ci jusqu'à l'ordre spécifié.

        Args:
            order (int): L'ordre maximal des polynômes d'Hermite à générer. Maximal 4

        Returns:
            dict: Un dictionnaire contenant les polynômes d'Hermite Hi.
        """

        mat1 = get_constant_mat1(self.Q)

        # Initialize the Non-dimensionnal Hermite polynomials : Hi := Hi/(r**2)
        self.Hi = {'0': mat1}

        # Initialiser Ci avec les composantes de base si ce n'est pas déjà fait
        # self.Ci doit contenir les Ci[a] pour a dans coords
        for n in range(1, self.order + 1):
            if n > MAX_ORDER:
                raise ValueError(
                    "L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

            indices_list = generate_indices(
                self.coords, n, permute=False, as_string=False)

            for indices in indices_list:
                key = ''.join(indices)

                # Compute delta_terms
                delta_terms = compute_delta_terms(indices)


                # Calculer le produit des Ci correspondants
                Ci_product = functools.reduce(
                    lambda x, y: sp.matrix_multiply_elementwise(x, y),
                    [self.Ci[idx] for idx in indices]
                )

                # Stocker Ci_product dans self.Ci
                self.Ci[key] = Ci_product

                # Calcul des polynômes d'Hermite Hi
                if n == 1:
                    self.Hi[key] = self.Ci[key]
                elif n == 2:
                    self.Hi[key] = self.Ci[key] - self.cs2 * kronecker(indices[0], indices[1]) * mat1
                elif n == 3:
                    dc_abc = compute_kronecker_third_order_coeff(indices, self.Ci)
                    self.Hi[key] = self.Ci[key] - self.cs2 * dc_abc
                elif n == 4:
                    dd_abcd, dc_abcd = compute_kronecker_fourth_order_coeff(indices, self.Ci)
                    self.Hi[key] = self.Ci[key] - self.cs2 * dc_abcd + self.cs2*self.cs2 * dd_abcd * mat1

      # if order >= 1:
      #     for a in coords:
      #         Hi[a] = self.Ci[a]

      # if order >= 2:
      #     indices = generate_indices(coords, 2, permute=False, as_string=False)
      #     for a, b in indices:
      #         self.Ci[a+b] = sp.matrix_multiply_elementwise(
      #             self.Ci[a], self.Ci[b])
      #         Hi[a+b] = self.Ci[a+b] - cs2*kronecker(a, b)*mat1

      # if order >= 3:
      #     indices = generate_indices(coords, 3, permute=False, as_string=False)
      #     for a, b, c in indices:
      #         self.Ci[a+b+c] = sp.matrix_multiply_elementwise(self.Ci[a+b], self.Ci[c])
      #         Hi[a+b+c] = self.Ci[a+b+c] - cs2*(kronecker(a, b)*self.Ci[c] + kronecker(
      #             a, c)*self.Ci[b] + kronecker(b, c)*self.Ci[a])

      # if order >= 4:
      #     indices = generate_indices(coords, 4, permute=False, as_string=False)
      #     for a, b, c, d in indices:
      #         d_ab = kronecker(a, b)
      #         d_ac = kronecker(a, c)
      #         d_ad = kronecker(a, d)
      #         d_bc = kronecker(b, c)
      #         d_bd = kronecker(b, d)
      #         d_cd = kronecker(c, d)
      #         dd_abcd = d_ab*d_cd+d_ac*d_bd+d_ad*d_bc

      #         dc_abcd_1 = d_ab*self.Ci[a+b] + d_ac * self.Ci[a+c] + d_ad*self.Ci[a+d]
      #         dc_abcd_2 = d_bc*self.Ci[b+c] + d_bd * self.Ci[b+d] + d_cd*self.Ci[c+d]
      #         dc_abcd = dc_abcd_1 + dc_abcd_2

      #         self.Ci[a+b+c+d] = sp.matrix_multiply_elementwise(
      #             self.Ci[a+b+c], self.Ci[d])

      #         Hi[a+b+c+d] = self.Ci[a+b+c+d] - cs2*dc_abcd+CS4*dd_abcd*mat1
      # if order >MAX_ORDER:
      #     raise ValueError("L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

    def _compute_rotated_hermite_polynomials(self):
        """
        Generate rotated third order Hermite polynomials.

        Returns:
            dict: A dictionary containing the rotated Hermite polynomials.
            The keys are '3r1', '3r2', '3r3', '3r4', '3r5', '3r6'.
            The values are the corresponding Hermite polynomials.
        """

        # List of required moments for rotated moments
        required_hi = ['xxy', 'yzz', 'xzz', 'xyy', 'yyz', 'xxz']

        # Initialize self.Hi if it doesn't exist
        if not hasattr(self, 'Hi'):
            self.Hi = {}

        # Check if the required Hermite polynomials exist
        missing_hi = [hi for hi in required_hi if hi not in self.Hi]

        if missing_hi:
            # Compute Malaspinas moments if any required moment is missing
            self._compute_hermite_polynomials()

        # Now compute the rotated Hermite polynomials
        self.Hi['3r1'] = self.Hi['xxy'] + self.Hi['yzz']
        self.Hi['3r2'] = self.Hi['xzz'] + self.Hi['xyy']
        self.Hi['3r3'] = self.Hi['yyz'] + self.Hi['xxz']
        self.Hi['3r4'] = self.Hi['xxy'] - self.Hi['yzz']
        self.Hi['3r5'] = self.Hi['xzz'] - self.Hi['xyy']
        self.Hi['3r6'] = self.Hi['yyz'] - self.Hi['xxz']

    def _compute_isotropic_hermite_polynomials(self, cs2, check=False):
        """
        Generate isotropic Hermite polynomials for the D3Q19 lattice.

        Returns:
            dict: A dictionary containing the isotropic Hermite polynomials.
            The keys are '4Dxyz', '4Dxzy', '4Dyzx', '4Ixyz', '4Ixzy', '4Iyzx'.
            The values are the corresponding Hermite polynomials.
        """

        if self.D != 3 or self.order < 4:
            raise ValueError(
                "Isotropic Hermite polynomials are only defined for 3D lattices (D=3) and are fourth-order (order >=4).")
        
        # List of required moments for istropic Hermite polynomials
        required_hi = ['xxyy', 'xxzz', 'yyzz', 'xx', 'yy', 'zz']

        # Initialize self.Hi if it doesn't exist
        if not hasattr(self, 'Hi'):
            self.Hi = {}

        # Check if the required Hermite polynomials exist
        missing_hi = [hi for hi in required_hi if hi not in self.Hi]

        if missing_hi:
            # Compute Malaspinas moments if any required moment is missing
            self._compute_hermite_polynomials()

        self.Hi['4Dxyz'] = self.Hi['xxyy'] + ONE_2*cs2*self.Hi['zz']
        self.Hi['4Dxzy'] = self.Hi['xxzz'] + ONE_2*cs2*self.Hi['yy']
        self.Hi['4Dyzx'] = self.Hi['yyzz'] + ONE_2*cs2*self.Hi['xx']

        # Isotropy correction for 4th order Hermite polynomials
        fourO9sqrtP = 4*ONE_9 * (3+2*SQRT3)
        fourO9sqrtM = 4*ONE_9 * (3-SQRT3)
        self.Hi['4Ixyz'] = fourO9sqrtP*self.Hi['4Dxyz'] + \
            fourO9sqrtM*(self.Hi['4Dxzy']+self.Hi['4Dyzx'])
        self.Hi['4Ixzy'] = fourO9sqrtP*self.Hi['4Dxzy'] + \
            fourO9sqrtM*(self.Hi['4Dxyz']+self.Hi['4Dyzx'])
        self.Hi['4Iyzx'] = fourO9sqrtP*self.Hi['4Dyzx'] + \
            fourO9sqrtM*(self.Hi['4Dxyz']+self.Hi['4Dxzy'])

        # Check isotropy
        if check:
            print('\nCheck isotropy of 4th order isotropic Hermite polynomes:')
            for index in ['4Ixyz', '4Ixzy', '4Iyzx']:
                Hi2 = sp.matrix_multiply_elementwise(
                    self.Hi[index], self.Hi[index])
                iso4 = sp.simplify(
                    np.sum(sp.matrix_multiply_elementwise(self.Wi, Hi2)))
                print(f'sum(w_i*Hi_{index}^2) = {iso4}. Should be {24*cs2**4}')
