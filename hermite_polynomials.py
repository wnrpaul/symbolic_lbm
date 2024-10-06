import sympy as sp
import functools
import logging
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

        # Initialize the non-dimensionnal Hermite polynomials : Hi := Hi/(r**2)
        self.Hi = {'0': mat1}
        logging.debug(f"Order 0: Hi['0'] = {print_mat(self.Hi['0'])}")

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
                logging.debug(f"Order {n}: Hi[{key}] = {print_mat(self.Hi[key])}")

    def _compute_rotated_hermite_polynomials(self):
        """
        Generate rotated third order Hermite polynomials.

        Returns:
            dict: A dictionary containing the rotated Hermite polynomials.
            The keys are '3r1', '3r2', '3r3', '3r4', '3r5', '3r6'.
            The values are the corresponding Hermite polynomials.
        """
        
        # Now compute the rotated Hermite polynomials
        self.Hi['3r1'] = self.Hi['xxy'] + self.Hi['yzz']
        logging.debug(f"Order 3: Hi['3r1'] = {print_mat(self.Hi['3r1'])}")
        self.Hi['3r2'] = self.Hi['xzz'] + self.Hi['xyy']
        logging.debug(f"Order 3: Hi['3r2'] = {print_mat(self.Hi['3r2'])}")
        self.Hi['3r3'] = self.Hi['yyz'] + self.Hi['xxz']
        logging.debug(f"Order 3: Hi['3r3'] = {print_mat(self.Hi['3r3'])}")
        self.Hi['3r4'] = self.Hi['xxy'] - self.Hi['yzz']
        logging.debug(f"Order 3: Hi['3r4'] = {print_mat(self.Hi['3r4'])}")
        self.Hi['3r5'] = self.Hi['xzz'] - self.Hi['xyy']
        logging.debug(f"Order 3: Hi['3r5'] = {print_mat(self.Hi['3r5'])}")
        self.Hi['3r6'] = self.Hi['yyz'] - self.Hi['xxz']
        logging.debug(f"Order 3: Hi['3r6'] = {print_mat(self.Hi['3r6'])}")

    def _compute_isotropic_hermite_polynomials(self, cs2):
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

        self.Hi['4Dxyz'] = self.Hi['xxyy'] + ONE_2*cs2*self.Hi['zz']
        logging.debug(f"Order 4: Hi['4Dxyz'] = {print_mat(self.Hi['4Dxyz'])}")
        self.Hi['4Dxzy'] = self.Hi['xxzz'] + ONE_2*cs2*self.Hi['yy']
        logging.debug(f"Order 4: Hi['4Dxzy'] = {print_mat(self.Hi['4Dxzy'])}")
        self.Hi['4Dyzx'] = self.Hi['yyzz'] + ONE_2*cs2*self.Hi['xx']
        logging.debug(f"Order 4: Hi['4Dyzx'] = {print_mat(self.Hi['4Dyzx'])}")

        # Isotropy correction for 4th order Hermite polynomials
        self.Hi['4Ixyz'] = ISO_4_ONE_9_3P2SQRT3*self.Hi['4Dxyz'] + \
            ISO_4_ONE_9_3MSQRT3*(self.Hi['4Dxzy']+self.Hi['4Dyzx'])
        logging.debug(f"Order 4: Hi['4Ixyz'] = {print_mat(self.Hi['4Ixyz'])}")
        self.Hi['4Ixzy'] = ISO_4_ONE_9_3P2SQRT3*self.Hi['4Dxzy'] + \
            ISO_4_ONE_9_3MSQRT3*(self.Hi['4Dxyz']+self.Hi['4Dyzx'])
        logging.debug(f"Order 4: Hi['4Ixzy'] = {print_mat(self.Hi['4Ixzy'])}")
        self.Hi['4Iyzx'] = ISO_4_ONE_9_3P2SQRT3*self.Hi['4Dyzx'] + \
            ISO_4_ONE_9_3MSQRT3*(self.Hi['4Dxyz']+self.Hi['4Dxzy'])
        logging.debug(f"Order 4: Hi['4Iyzx'] = {print_mat(self.Hi['4Iyzx'])}")
