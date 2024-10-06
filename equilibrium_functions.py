#!/usr/bin/env python3
"""
Module `equilibrium_functions.py`

Ce module contient les définitions des fonctions d'équilibre pour la méthode de Boltzmann sur réseau (LBM).
Il permet de calculer les fonctions d'équilibre en fonction de différents paramètres et configurations.

Classes :
- EquilibriumFunction : classe de base pour les fonctions d'équilibre.
- GuoImproved : implémentation spécifique de la fonction d'équilibre améliorée de Guo.

Exemple d'utilisation :
>>> eq_func = GuoImproved(D=3, Q=19, is_thermal=True, order_0=4)
>>> eq_func.compute()
"""
import logging
import functools
import operator
import sympy as sp
import numpy as np
from lattices import Lattice
from symbolic_lbm.hermite_polynomials import HermitePolynomials
from symbols_mapping import DEFAULT_SYMBOLS
from utils import *
from constants import *


class EquilibriumFunction:
    def __init__(self, D=3, Q=19, is_thermal=False, order_0=3, symbols=None):
        self.D = D
        self.Q = Q
        self.is_thermal = is_thermal
        self.order_0 = order_0
        self.symbols = symbols  # Dictionnaire de symboles personnalisés
        self.name = f'D{self.D}Q{self.Q}_T{int(self.is_thermal)}_O{self.order_0}'
        logging.info(f"Initializing equilibrium function {self.name}")

        if symbols is None:
            self.symbols = {}

        # Fusionner les symboles par défaut avec ceux de l'utilisateur
        self.symbols = {**DEFAULT_SYMBOLS, **self.symbols}
        logging.info(f"Using symbols: {self.symbols}")
        self.coords = COORDS[:self.D]

        # for sym in required_symbols:
        #     if sym not in self.symbols:
        #         warnings.warn(f"Le symbole '{sym}' est requis mais n'a pas été défini, valeur par défaut utilisée.")
        #         self.symbols[sym] = sym  # Utilisation du nom du symbole comme valeur par défaut

        # Vérifier que tous les symboles requis sont présents
        required_symbols = ['rho', 'tp', 'cs'] + [f'u{coord}' for coord in self.coords] + [
            f'f{coord}' for coord in self.coords]
        for sym in required_symbols:
            if sym not in self.symbols:
                raise ValueError(
                    f"Le symbole '{sym}' est requis mais n'a pas été défini.")

            # Fusionner les symboles par défaut avec ceux de l'utilisateur
            # self.symbols = DEFAULT_SYMBOLS.copy()
            # if symbols:
            #   for key, value in symbols.items():
            #     if key in self.symbols:
            #         self.symbols[key] = value
            #     else:
            #         warnings.warn(f"Symbole inconnu '{key}' fourni. Il sera ignoré.")

        # Initialisation du réseau
        lattice = Lattice(self.D, self.Q)
        self.Wi = lattice.Wi
        self.Ci = lattice.Ci

        # Initialisation des variables macroscopiques
        self._initialize_macroscopic_vars()

        
    def _check_lattice_isotropy(self):
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"Checking lattice isotropy for D{self.D}Q{self.Q}:")
            isotropy_0 = np.sum(self.Wi)
            logging.debug(f"Order 0: sum(wi) = {isotropy_0}. Should be 1")
            
            for a in COORDS[:self.D]:
                isotropy = np.sum(sp.matrix_multiply_elementwise(self.Wi, self.Ci[a]))
                logging.debug(f"Order 1: sum(Wi*Ci{a}) = {isotropy}. Should be 0")

            for indices in generate_indices(COORDS[:self.D], 2, permute=False, as_string=False):
                key = ''.join(indices)
                isotropy = np.sum(sp.matrix_multiply_elementwise(self.Wi, self.Ci[key]))
                logging.debug(f"Order 2: sum(Wi*Ci{key}) = {isotropy}. Should be {self.cs2*kronecker(key[0], key[1])}")

            for indices in generate_indices(COORDS[:self.D], 3, permute=False, as_string=False):
                key = ''.join(indices)
                isotropy = np.sum(sp.matrix_multiply_elementwise(self.Wi, self.Ci[key]))
                logging.debug(f"Order 3: sum(Wi*Ci{key}) = {isotropy}. Should be 0")

            for indices in generate_indices(COORDS[:self.D], 4, permute=False, as_string=False):
                key = ''.join(indices)
                isotropy = np.sum(sp.matrix_multiply_elementwise(self.Wi, self.Ci[key]))
                sol = self.cs4*compute_kronecker_fourth_order(indices)
                logging.debug(f"Order 4: sum(Wi*Ci{key}) = {isotropy}. Should be {sol}")

    def summary(self):
        """
        Prints a summary of important variables of the object.
        """
        print("\n===== Equilibrium Summary =====")
        print(f"Lattice   : D{self.D}Q{self.Q}")
        print(f"Isothermal: {'Yes' if not self.is_thermal else 'No'}")
        print(f"Order     : {self.order_0}")
        print(f"\n")
        print(f"Macroscopic variable symbols:")
        print(f"  - rho: {self.rho}")
        print(f"  - ux: {self.ux}")
        print(f"  - uy: {self.uy}")
        print(f"  - uz: {self.uz}") if self.D == 3 else None
        print(f"  - tp: {self.tp}")
        print(f"Number of added Hermite polynomials: {self.nb_hermite_polynomials}")
        print("=================================\n")

    def _initialize_macroscopic_vars(self):
        # Récupération des noms de symboles à partir du dictionnaire
        # sqrt(r Tref), Lattice sound speed
        cs_symbol = self.symbols.get('cs', 'cs')
        rho_symbol = self.symbols.get('rho', 'rho')
        tp_symbol = self.symbols.get('tp', 'tp')
        ux_symbol = self.symbols.get('ux', 'ux')
        uy_symbol = self.symbols.get('uy', 'uy')
        uz_symbol = self.symbols.get('uz', 'uz')
        fx_symbol = self.symbols.get('fx', 'fx')
        fy_symbol = self.symbols.get('fy', 'fy')
        fz_symbol = self.symbols.get('fz', 'fz')
        ups_symbol = self.symbols.get('ups', 'ups')

        self.cs_dim = sp.symbols(cs_symbol, real=True)
        # Standard quadrature lattice coefficient
        self.r_qua = self.cs_dim*sp.sqrt(3)
        self.cs = self.cs_dim/self.r_qua       # Adimensionnal lattice unit
        self.cs2 = self.cs*self.cs
        self.cs4 = self.cs2*self.cs2
        self.cs8 = self.cs4*self.cs4
        self.one_on_facto_cs2n = compute_one_on_factorial_cs2_n(
            self.order_0, self.cs2)

        # Création des variables symboliques
        self.rho = sp.symbols(rho_symbol, real=True)
        self.ux = sp.symbols(ux_symbol, real=True)
        self.uy = sp.symbols(uy_symbol, real=True)
        self.uz = sp.symbols(uz_symbol, real=True)
        self.fx = sp.symbols(fx_symbol, real=True)
        self.fy = sp.symbols(fy_symbol, real=True)
        self.fz = sp.symbols(fz_symbol, real=True)
        self.tp = sp.symbols(tp_symbol, real=True) if self.is_thermal else 1
        self.ups = sp.symbols(ups_symbol, real=True)

        # Initialisation des vecteurs de vitesse et d'accélération
        self.uvec = {}  # Dictionnaire pour les vitesses

        # Initialisation des polynômes d'Hermite
        self.nb_hermite_polynomials = 0
        self.Hi = {}

        for coord in self.coords:
            u_symbol = self.symbols.get(f'u{coord}', f'u{coord}')

            self.uvec[coord] = sp.symbols(u_symbol, real=True)
            # Autres initialisations nécessaires

    def _compute_malaspinas_moments(self):
        self.a0_malaspinas = {'0': self.rho}
        logging.debug(f"Order 0: a0_malaspinas[0] = {self.rho}")

        # Boucle sur les ordres de 1 à order_0
        for order in range(1, self.order_0 + 1):
            if order > MAX_ORDER:
                raise ValueError(
                    "L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

            # Générer les indices pour l'ordre actuel
            indices_list = generate_indices(
                self.coords, order, permute=False, as_string=False)
            for indices in indices_list:
                key = ''.join(indices)
                # Calculer le produit des vitesses pour les indices donnés
                u_product = functools.reduce(
                    operator.mul, [self.uvec[ind] for ind in indices])

                # Stocker le produit dans self.uvec s'il n'existe pas déjà
                if key not in self.uvec:
                    self.uvec[key] = u_product

                logging.debug(f"Order {order}: a0_malaspinas[{key}] = {self.rho} * {self.uvec[key]}")
                self.a0_malaspinas[key] = self.rho * self.uvec[key]

    def _compute_hermite_moments(self):
        self._compute_malaspinas_moments()
        self.a0_hermite = self.a0_malaspinas.copy()
        logging.info(f"Order 0: a0_hermite[0] = {self.a0_malaspinas['0']}")

        self.p_h = self.rho * self.cs2 * (self.tp - 1)
        self.q_h = self.rho * (self.cs2 * (self.tp - 1)) ** 2

        # Boucle sur les ordres de 1 à order_0
        for order in range(2, self.order_0 + 1):
            if order > MAX_ORDER:
                raise ValueError(
                    "L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

            # Générer les indices pour l'ordre actuel
            indices_list = generate_indices(
                self.coords, order, permute=False, as_string=False)
            for indices in indices_list:
                key = ''.join(indices)

                if order == 2:
                    self.a0_hermite[key] += self.p_h * \
                        kronecker(indices[0], indices[1])
                elif order == 3:
                    du_abc = compute_kronecker_third_order_coeff(
                        indices, self.uvec)
                    self.a0_hermite[key] += self.p_h * du_abc
                elif order == 4:
                    dd_abcd, du_abcd = compute_kronecker_fourth_order_coeff(
                        indices, self.uvec)
                    self.a0_hermite[key] += self.p_h*du_abcd+self.q_h*dd_abcd
                logging.debug(f"Order {order}: a0_hermite[{key}] = {self.a0_hermite[key]}")

    def _compute_rotated_malaspinas_moments(self):
        """
        Generate rotated third order Hermite moments.

        Returns:
            dict: A dictionary containing the rotated Hermite moments.
            The keys are '3r1', '3r2', '3r3', '3r4', '3r5', '3r6'.
            The values are the corresponding Hermite moments.
        """

        if self.D != 3 or self.order_0 < 3:
            raise ValueError(
                "Rotated Hermite moments are only defined for 3D lattices (D=3) and are third-order (order >=3).")
        
        # Now compute the rotated moments
        self.a0_malaspinas['3r1'] = 3 * (self.a0_malaspinas['xxy'] + self.a0_malaspinas['yzz'])
        logging.debug(f"Order 3: a0_malaspinas[3r1]: {self.a0_malaspinas['3r1']}")

        self.a0_malaspinas['3r2'] = 3 * (self.a0_malaspinas['xzz'] + self.a0_malaspinas['xyy'])
        logging.debug(f"Order 3: a0_malaspinas[3r2]: {self.a0_malaspinas['3r2']}")

        self.a0_malaspinas['3r3'] = 3 * (self.a0_malaspinas['yyz'] + self.a0_malaspinas['xxz'])
        logging.debug(f"Order 3: a0_malaspinas[3r3]: {self.a0_malaspinas['3r3']}")

        self.a0_malaspinas['3r4'] = self.a0_malaspinas['xxy'] - self.a0_malaspinas['yzz']
        logging.debug(f"Order 3: a0_malaspinas[3r4]: {self.a0_malaspinas['3r4']}")

        self.a0_malaspinas['3r5'] = self.a0_malaspinas['xzz'] - self.a0_malaspinas['xyy']
        logging.debug(f"Order 3: a0_malaspinas[3r5]: {self.a0_malaspinas['3r5']}")

        self.a0_malaspinas['3r6'] = self.a0_malaspinas['yyz'] - self.a0_malaspinas['xxz']
        logging.debug(f"Order 3: a0_malaspinas[3r6]: {self.a0_malaspinas['3r6']}")
        

    def _multiply_by_weight_and_simplify(self, f_eq):
        logging.info(f'Number of added Hermite polynomials : {self.nb_hermite_polynomials}')
        f_eq = sp.matrix_multiply_elementwise(self.Wi, f_eq)
        f_eq = sp.simplify(f_eq)
        logging.info(f'{self.name} equilibrium function:\n{sp.pretty(f_eq)}')
        logging.debug(f'f_eq = {print_mat(f_eq)}')
        return f_eq

    def _initizalize_feq_order_0(self):
        # Initialisation = order 0
        factor = self.one_on_facto_cs2n[0]
        f_eq = factor * self.a0_hermite['0']*self.Hi['0']
        logging.info(f'Order 0: f_eq += {factor}*a0_hermite[0]*Hi[0]')
        logging.debug(f'Order 0: f_eq += {print_mat(f_eq)}')
        self.nb_hermite_polynomials = 1
        return f_eq

    def _compute_grad_hermite_terms_by_order(self, f_eq, order):
        """
        Computes the Grad-Hermite series expansion terms for a given order and updates the equilibrium distribution function. We sum over all the indexes of the Hermite tensor.
        The equilibrium function is updated as follows:
       
        feq_i = W_i * sum_n [ 1/(n! * cs^(2n)) * a0_hermite_n[key] : Hi_n[key] 
        
        Ex: The loop for the 2nd order tensor is: key={xx, xy, xz, yx, yy, yz, zx, zy, zz}.
        Since Hi is symmetric, we only need to compute the upper triangular part of the tensor.
        To do so, we use the generate_indices function with permute=False and multiply the result by the multiplicity of the indices. (Another method would be to simply set permute=True in generate_indices, but this would be less efficient for high order polynomials).

        Args:
            f_eq (float): The equilibrium distribution function.
            order (int): The order of the Hermite terms.

        Returns:
            float: The updated equilibrium distribution function.
        """
        indices_list = generate_indices(
            self.coords, order, permute=False, as_string=False)
        for indices in indices_list:
            key = ''.join(indices)
            if self.Hi[key].is_zero_matrix:
                logging.info(f'Order {order}: Hi[{key}] = [0]_{self.Q}')
                continue

            factor = self.one_on_facto_cs2n[order]
            
            # Calculate multiplicity of the indices
            # multiplicty = n! / (n1! * n2! * ... * nk!)
            # Ex: For order 2, the multiplicity of 'xx' is 1, 'xy' is 2
            # By calculating and applying the multiplicity, we ensure that 'xy'
            # is counted twice in the sum. This approach is more efficient than
            # changing permute to True in generate_indices. Especially for high
            # order polynomials.
            multiplicity = compute_multiplicity(order, indices)

            term = factor * multiplicity * self.a0_hermite[key] * self.Hi[key]
            f_eq += term

            self.nb_hermite_polynomials += 1

            logging.info(f'Order {order}: f_eq += {multiplicity} * {factor} * a0_hermite[{key}] * Hi[{key}]')
            logging.debug(f'Order {order}: f_eq += {print_mat(term)}')
        return f_eq
    
    def _get_hermite_calculator(self):
        # Compute the Hermite polynomials of interests
        hermiteCalculator = HermitePolynomials(self.Ci, self.cs2,
                                               self.order_0, self.Q, self.D)
        
        # Check lattice isotropy if logging.DEBUG is enabled
        self._check_lattice_isotropy()

        return hermiteCalculator
    
        

    def compute_feq(self):
        # Hermite series expansion of the distribution functions by order
        # # We sum over all the indexes of the tensor Hi:
        # Ex: The loop for the 2nd order tensor is: xx, xy, xz, yx, yy, yz, zx, zy, zz
        #
        # Construction of the equilibrium function based on an expansion over the   Hermite basis:
        #
        # f0_i = w_i * sum_n [ 1/(n! * cs^(2n)) * a0_H(n) : Hi(n) ]

        # Compute the Hermite moments
        self._compute_hermite_moments()

        # Compute the Hermite polynomials
        hermiteCalculator = self._get_hermite_calculator()
        self.Hi = hermiteCalculator.Hi

        # Initialisation = order 0
        f_eq = self._initizalize_feq_order_0()

        return f_eq


class GradHermite(EquilibriumFunction):
    def __init__(self, D=3, Q=19, is_thermal=False, order_0=3, symbols=None):
        super().__init__(D, Q, is_thermal, order_0, symbols)
        self.name = self.name+'_grad-hermite'

    def compute_feq(self):
        f_eq = super().compute_feq()  # Appelle la version de bas

        # Boucle sur les ordres
        for order in range(1, self.order_0+1):
            f_eq = self._compute_grad_hermite_terms_by_order(f_eq, order)

        return self._multiply_by_weight_and_simplify(f_eq)


class RotatedBaseEquilibrium(EquilibriumFunction):
    def __init__(self, D, Q, is_thermal, order_0, iso_terms=False, symbols=None):
        super().__init__(D, Q, is_thermal, order_0, symbols)
        self.iso_terms = iso_terms
        # Initialisations communes supplémentaires si nécessaire

    def _get_hermite_calculator(self):
        hermiteCalculator = super()._get_hermite_calculator()
        hermiteCalculator._compute_rotated_hermite_polynomials()
        
        if self.iso_terms:
            hermiteCalculator._compute_isotropic_hermite_polynomials(self.cs2)
            
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                self.Hi = hermiteCalculator.Hi
                # Check isotropy
                logging.debug('Check isotropy of fourth order isotropic Hermite polynomes:')
                for index in ['4Ixyz', '4Ixzy', '4Iyzx']:
                    Hi2 = sp.matrix_multiply_elementwise(
                        self.Hi[index], self.Hi[index])
                    iso4 = sp.simplify(
                        np.sum(sp.matrix_multiply_elementwise(self.Wi, Hi2)))
                    logging.debug(f'sum(w_i*Hi_{index}^2) = {iso4}. Should be {24*self.cs8}')

        return hermiteCalculator

    def _compute_fourth_order_terms(self, f_eq):
        """
        Méthode à surcharger dans les sous-classes pour le calcul spécifique de l'ordre 4.
        """
        return f_eq

    def compute_feq(self):
        f_eq = super().compute_feq()  # Appelle la version de base

        # Compute the rotated Hermite moments
        self._compute_rotated_malaspinas_moments()

        # Add order 1 and 2 terms only
        f_eq = self._compute_grad_hermite_terms_by_order(f_eq, 1)
        f_eq = self._compute_grad_hermite_terms_by_order(f_eq, 2)

        # Add rotated order 3 terms
        for index in ['3r1', '3r2', '3r3', '3r4', '3r5', '3r6']:
            factor = self.one_on_facto_cs2n[3]
            f_eq += factor * self.a0_malaspinas[index] * self.Hi[index]
            logging.info(f'Order 3: f_eq += {factor} * a0_malaspinas[{index}] * Hi[{index}]')
            logging.debug(f'Order 3: f_eq += {print_mat(factor * self.a0_malaspinas[index] * self.Hi[index])}')
            self.nb_hermite_polynomials += 1

        f_eq = self._compute_fourth_order_terms(f_eq)
        
        return f_eq


class D3Q19GuoImproved(RotatedBaseEquilibrium):
    def __init__(self, symbols=None):
        D = 3
        Q = 19
        is_thermal = True
        order_0 = 4
        iso_terms = True
        super().__init__(D, Q, is_thermal, order_0, iso_terms, symbols)
        self.name = self.name+'_guo-improved'
        # Autres initialisations spécifiques à GuoImproved_D3Q19

    def _compute_fourth_order_terms(self, f_eq):
        # Implémentation spécifique pour GuoImproved_D3Q19
        for index in ['4Ixyz', '4Ixzy', '4Iyzx']:
            self.a0_hermite[index] = -2 * self.rho * self.cs4 * (self.tp - 1)
            factor = self.one_on_facto_cs2n[4]
            f_eq += factor * self.a0_hermite[index] * self.Hi[index]
            logging.info(f'Order 4: f_eq += {factor} * a0_hermite[{index}] * Hi[{index}]')
            logging.debug(f'Order 4: f_eq += {print_mat(factor * self.a0_hermite[index] * self.Hi[index])}')
            self.nb_hermite_polynomials += 1

        return f_eq

    def compute_feq(self):
        f_eq = super().compute_feq()
        return self._multiply_by_weight_and_simplify(f_eq)


class D3Q19Iso(RotatedBaseEquilibrium):
    def __init__(self, version=1, symbols=None):
        D = 3
        Q = 19
        is_thermal = True
        order_0 = 4
        iso_terms = True
        super().__init__(D, Q, is_thermal, order_0, iso_terms, symbols)
        self.version = version  # 1 or 2
        self.name = self.name+'_iso-v'+str(version)

    def _compute_fourth_order_terms(self, f_eq):
        # Implémentation spécifique pour IsoV1
        circu = generate_circular_permutation(['zz', 'yy', 'xx'])
        for ind, index in enumerate(['4Ixyz', '4Ixzy', '4Iyzx']):
            if self.version == 1:
                iso_coef = self.rho * self.cs4 * (1 - self.tp)
            elif self.version == 2:
                iso_coef = self.rho * self.cs4 * (self.tp+1)
            else:
                raise ValueError("Version must be 1 or 2")
            logging.debug(f'Order 4: iso coeff = {iso_coef}')


            a_Ai = ISO_4_ONE_9_3P2SQRT3 * (iso_coef + self.cs2 * ONE_2 * self.a0_hermite[circu[ind][0]])
            a_Bi = ISO_4_ONE_9_3MSQRT3 * (iso_coef + self.cs2 * ONE_2 * self.a0_hermite[circu[ind][1]])
            a_Ci = ISO_4_ONE_9_3MSQRT3 * (iso_coef + self.cs2 * ONE_2 * self.a0_hermite[circu[ind][2]])
            logging.debug(f'Order 4: a_Ai[{index}] = {a_Ai}')
            logging.debug(f'Order 4: a_Bi[{index}] = {a_Bi}')
            logging.debug(f'Order 4: a_Ci[{index}] = {a_Ci}')
            self.a0_hermite[index] = a_Ai + a_Bi + a_Ci
            logging.debug(f'Order 4: a0_hermite[{index}] = {self.a0_hermite[index]}')
            factor = self.one_on_facto_cs2n[4]
            f_eq += factor * self.a0_hermite[index] * self.Hi[index]
            logging.info(f'Order 4: f_eq += {factor} * a0_H[{index}] * Hi[{index}]')
            logging.debug(f'Order 4: f_eq += {print_mat(factor * self.a0_hermite[index] * self.Hi[index])}')
            self.nb_hermite_polynomials += 1
        
        return f_eq

    def compute_feq(self):
        f_eq = super().compute_feq()
        return self._multiply_by_weight_and_simplify(f_eq)


class D3Q19Unified(RotatedBaseEquilibrium):
    def __init__(self, symbols=None):
        D = 3
        Q = 19
        is_thermal = True
        order_0 = 3
        iso_terms = False
        super().__init__(D, Q, is_thermal, order_0, iso_terms, symbols)
        self.name = self.name+'_unified'
        # Autres initialisations spécifiques à Unified

    def _compute_hermite_moments(self):
        super()._compute_hermite_moments()  # Appelle la version de base

        # Only modify order 2 to include Upsilon term
        indices_list = generate_indices(
            self.coords, 2, permute=False, as_string=False)
        for indices in indices_list:
            key = ''.join(indices)
            self.a0_hermite[key] = self.a0_malaspinas[key] + \
                self.ups * self.p_h * kronecker(indices[0], indices[1])
            logging.debug(f"Order 2: a0_hermite[{key}] = {self.a0_hermite[key]}")
            
    
    def compute_feq(self):
        f_eq = super().compute_feq()  # Appelle la version de base

        # Add d0i temperature term
        d0i = get_constant_d0i(self.Q)
        fu_factor = matrix_divide_elementwise(self.Wi-d0i, self.Wi)
        term = self.one_on_facto_cs2n[0] * sp.matrix_multiply_elementwise(fu_factor, self.Hi['0'])*self.a0_hermite['0']*(self.tp-1)*(1-self.ups)
        f_eq += term
        logging.info(f'Order 0: f_eq += {self.one_on_facto_cs2n[0]} * (Wi - d0i) * a0_hermite[0] * Hi[0] * ({self.tp}-1) * (1-{self.ups})')
        logging.debug(f'Order 0: f_eq += {term}')

        return self._multiply_by_weight_and_simplify(f_eq)
