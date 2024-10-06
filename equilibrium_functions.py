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
import warnings
import functools
import operator
import sympy as sp
from lattices import Lattice
from symbolic_lbm.hermite_polynomials import HermitePolynomials
from symbols_mapping import DEFAULT_SYMBOLS
from utils import *
from constants import *


class EquilibriumFunction:
    def __init__(self, D=3, Q=19, is_thermal=0, order_0=3, symbols=None):
        self.D = D
        self.Q = Q
        self.is_thermal = is_thermal
        self.order_0 = order_0
        self.symbols = symbols  # Dictionnaire de symboles personnalisés

        if symbols is None:
            self.symbols = {}

        # Fusionner les symboles par défaut avec ceux de l'utilisateur
        self.symbols = {**DEFAULT_SYMBOLS, **self.symbols}
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

        # Initialisation des variables communes
        self.mat0 = get_constant_mat0(self.Q)
        self.mat1 = get_constant_mat1(self.Q)

        # Initialisation des variables macroscopiques
        self._initialize_macroscopic_vars()

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
        self.one_on_facto_cs2n = compute_one_on_factorial_cs2_n(self.order_0, self.cs2)

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

        # Calcul des autres variables en fonction des symboles
        #   self.p0 = self.rho * self.cs2 * self.tp
        #   self.q0 = self.rho * (self.cs2 * self.tp) ** 2
        #   self.p_h = self.rho * self.cs2 * (self.tp - 1)
        #   self.q_h = self.rho * (self.cs2 * (self.tp - 1)) ** 2

        #   # Use for the fictitious third order moment (@nguyen2022)
        #   self.p_star = self.rho*self.cs2

        # Initialisation des vecteurs de vitesse et d'accélération
        self.uvec = {}  # Dictionnaire pour les vitesses
        self.fvec = {}  # Dictionnaire pour les forces

        # Initialisation des polynômes d'Hermite
        self.nb_hermite_polynomials = 0
        self.Hi = {}

        coords = ('x', 'y', 'z')[:self.D]
        for coord in coords:
            u_symbol = self.symbols.get(f'u{coord}', f'u{coord}')
            f_symbol = self.symbols.get(f'f{coord}', f'f{coord}')

            self.uvec[coord] = sp.symbols(u_symbol, real=True)
            self.fvec[coord] = sp.symbols(f_symbol, real=True)

            # Autres initialisations nécessaires

    def _compute_malaspinas_moments(self):
        # Malaspinas moments: rho*[u](n)
        self.a0_malaspinas = {'0': self.rho}

        coords = COORDS[:self.D]

        # Boucle sur les ordres de 1 à order_0
        for order in range(1, self.order_0 + 1):
            if order > MAX_ORDER:
                raise ValueError(
                    "L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

            # Générer les indices pour l'ordre actuel
            indices_list = generate_indices(
                coords, order, permute=False, as_string=False)
            for indices in indices_list:
                key = ''.join(indices)
                # Calculer le produit des vitesses pour les indices donnés
                u_product = functools.reduce(
                    operator.mul, [self.uvec[coord] for coord in indices])

                # Stocker le produit dans self.uvec s'il n'existe pas déjà
                if key not in self.uvec:
                    self.uvec[key] = u_product

                # Calculer le moment de Malaspinas correspondant
                self.a0_malaspinas[key] = self.rho * self.uvec[key]

    def _compute_hermite_moments(self):
        self._compute_malaspinas_moments()
        # Initialisation des moments d'Hermite
        self.a0_hermite = self.a0_malaspinas

        coords = COORDS[:self.D]

        p_h = self.rho * self.cs2 * (self.tp - 1)
        q_h = self.rho * (self.cs2 * (self.tp - 1)) ** 2

        # Boucle sur les ordres de 1 à order_0
        for order in range(2, self.order_0 + 1):
            if order > MAX_ORDER:
                raise ValueError(
                    "L'ordre maximal des polynômes d'Hermite est 4 pour le moment.")

            # Générer les indices pour l'ordre actuel
            indices_list = generate_indices(
                coords, order, permute=False, as_string=False)
            for indices in indices_list:
                key = ''.join(indices)

                if order == 2:
                    self.a0_hermite[key] += p_h * \
                        kronecker(indices[0], indices[1])
                elif order == 3:
                    du_abc = compute_kronecker_third_order_coeff(
                        indices, self.uvec)
                    self.a0_hermite[key] += p_h * du_abc
                elif order == 4:
                    dd_abcd, du_abcd = compute_kronecker_fourth_order_coeff(
                        indices, self.uvec)
                    self.a0_hermite[key] += p_h*du_abcd+q_h*dd_abcd


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
        
        # List of required moments for rotated moments
        required_a0 = ['xxy', 'yzz', 'xzz', 'xyy', 'yyz', 'xxz']

        # Initialize self.a0_malaspinas if it doesn't exist
        if not hasattr(self, 'a0_malaspinas'):
            self.a0_malaspinas = {}

        # Check if the required moments exist
        missing_a0 = [a0 for a0 in required_a0 if a0 not in self.a0_malaspinas]

        if missing_a0:
            # Compute Malaspinas moments if any required moment is missing
            self._compute_malaspinas_moments()
        
        # Now compute the rotated moments
        self.a0_malaspinas['3r1'] = 3*(self.a0_malaspinas['xxy'] + self.a0_malaspinas['yzz'])
        self.a0_malaspinas['3r2'] = 3*(self.a0_malaspinas['xzz'] + self.a0_malaspinas['xyy'])
        self.a0_malaspinas['3r3'] = 3*(self.a0_malaspinas['yyz'] + self.a0_malaspinas['xxz'])
        self.a0_malaspinas['3r4'] = self.a0_malaspinas['xxy'] - self.a0_malaspinas['yzz']
        self.a0_malaspinas['3r5'] = self.a0_malaspinas['xzz'] - self.a0_malaspinas['xyy']
        self.a0_malaspinas['3r6'] = self.a0_malaspinas['yyz'] - self.a0_malaspinas['xxz']
       
    def _multiply_by_weight_and_simplify(self, f_eq):
        f_eq = sp.matrix_multiply_elementwise(self.Wi, f_eq)
        f_eq = sp.simplify(f_eq)
        return f_eq

    def _initizalize_feq_order_0(self):
        # Initialisation = order 0
        f_eq = self.one_on_facto_cs2n[0] * self.a0_hermite['0']*self.Hi['0']
        print(f'Order 0: f_eq += {self.one_on_facto_cs2n[0]}*a0_hermite[0]*Hi[0]')
        self.nb_hermite_polynomials = 1
        return f_eq
    
    def _compute_feq_grad_hermite_terms_by_order(self, f_eq, order):
        indices_list = generate_indices(self.coords, order, permute=False, as_string=False)
        for indices in indices_list:
            key = ''.join(indices)
            factor = self.one_on_facto_cs2n[order]
            term = factor * self.a0_hermite[key] * self.Hi[key]
            f_eq += term
            self.nb_hermite_polynomials += 1
            print(f'Order {order}: f_eq += {factor}*a0_hermite[{key}]*Hi[{key}]')
        return f_eq


    def compute(self):
        raise NotImplementedError(
            "Cette méthode doit être implémentée par les sous-classes.")


class RotatedBaseEquilibrium(EquilibriumFunction):
    def __init__(self, D, Q, is_thermal, order_0, symbols=None):
        super().__init__(D, Q, is_thermal, order_0, symbols)
        # Initialisations communes supplémentaires si nécessaire

    def compute(self):
        # Get Hermite moments
        self._compute_hermite_moments()
        self._compute_rotated_malaspinas_moments()

        # Get Hermite polynomials
        hermites = HermitePolynomials(self.Ci, self.cs2, self.order_0, self.Q, self.D)
        hermites._compute_rotated_hermite_polynomials()
        hermites._compute_isotropic_hermite_polynomials(self.cs2)
        self.Hi = hermites.Hi

        # Initialisation = order 0
        f_eq = self._initizalize_feq_order_0()

        f_eq = self._compute_feq_grad_hermite_terms_by_order(f_eq, 1)
        f_eq = self._compute_feq_grad_hermite_terms_by_order(f_eq, 2)

        # Ordre 3
        for index in ['3r1', '3r2', '3r3', '3r4', '3r5', '3r6']:
            factor = self.one_on_facto_cs2n[3]
            f_eq += factor * self.a0_malaspinas[index] * self.Hi[index]
            print(f'Order 3. Rotated Hermite Basis: f_eq += {factor}*a0_hermite[{index}]*Hi[{index}]')
            self.nb_hermite_polynomials += 1

        self._compute_order_4_terms(f_eq)
            
        print(f'\nNumber of added orthogonal Hermite polynomials : {self.nb_hermite_polynomials}')
        return self._multiply_by_weight_and_simplify(f_eq)
    
    def _compute_order_4_terms(self, f_eq):
        """
        Méthode à surcharger dans les sous-classes pour le calcul spécifique de l'ordre 4.
        """
        raise NotImplementedError("La méthode _compute_order_4_terms doit être surchargée dans la sous-classe.")


class D3Q19GuoImproved(RotatedBaseEquilibrium):
    def __init__(self, symbols=None):
        D = 3
        Q = 19
        is_thermal = 1
        order_0 = 4
        super().__init__(D, Q, is_thermal, order_0, symbols)
        # Autres initialisations spécifiques à GuoImproved_D3Q19

    def _compute_order_4_terms(self, f_eq):
        # Implémentation spécifique pour GuoImproved_D3Q19
        for index in ['4Ixyz', '4Ixzy', '4Iyzx']:
            self.a0_hermite[index] = -2 * self.rho * self.cs4 * (self.tp - 1)
            factor = self.one_on_facto_cs2n[4]
            f_eq += factor * self.a0_hermite[index] * self.Hi[index]
            print(f'Order 4: Istropy: f_eq += {factor} * a0_H[{index}] * Hi[{index}] (Isotropy)')
            self.nb_hermite_polynomials += 1


class D3Q19Iso(RotatedBaseEquilibrium):
    def __init__(self, version=1, symbols=None):
        D = 3
        Q = 19
        is_thermal = 1
        order_0 = 4
        super().__init__(D, Q, is_thermal, order_0, symbols)
        self.version = version # 1 or 2

    def _compute_order_4_terms(self, f_eq):
        # Implémentation spécifique pour IsoV1
        circu = circular_permutation(['zz', 'yy', 'xx'])
        for ind, index in enumerate(['4Ixyz', '4Ixzy', '4Iyzx']):
            if self.version == 1:
                iso_coef = self.rho * self.cs4 * (1 - self.tp)
            elif self.version == 2:
                iso_coef = self.rho * self.cs4 * (self.tp+1)

            a_Ai = 4*ONE_9*(3+2*SQRT3) * (iso_coef + self.cs2*ONE_2 * self.a0_hermite[circu[ind][0]])
            a_Bi = 4*ONE_9*(3+2*SQRT3) * (iso_coef + self.cs2*ONE_2 * self.a0_hermite[circu[ind][1]])
            a_Ci = 4*ONE_9*(3-SQRT3) * (iso_coef + self.cs2*ONE_2 * self.a0_hermite[circu[ind][2]])
            self.a0_hermite[index] = a_Ai + a_Bi + a_Ci
            factor = self.one_on_facto_cs2n[4]
            f_eq += factor * self.a0_hermite[index] * self.Hi[index]
            print(f'Order 4: Istropy: f_eq += {factor} * a0_H[{index}] * Hi[{index}]')
            self.nb_hermite_polynomials += 1


class GradHermite(EquilibriumFunction):
    def __init__(self, D=3, Q=19, is_thermal=0, order_0=3, symbols=None):
        super().__init__(D, Q, is_thermal, order_0, symbols)

    def compute(self):
        # Hermite series expansion of the distribution functions by order
        # We sum over all the indexes of the tensor Hi: 
        # Ex: The loop for the 2nd order tensor is: xx, xy, xz, yx, yy, yz, zx, zy, zz
        #
        # Construction of the equilibrium function based on an expansion over the Hermite basis:
        #
        # f0_i = w_i * sum_n [ 1/(n! * cs^(2n)) * a0_H(n) : Hi(n) ]

        # Get Hermite moments
        self._compute_hermite_moments()

        # Get Hermite polynomials
        hermites = HermitePolynomials(self.Ci, self.cs2, self.order_0, self.Q, self.D)
        Hi = hermites.Hi

        # Initialisation = order 0
        f_eq = self._initizalize_feq_order_0()

        # Boucle sur les ordres
        for order in range(1, self.order_0+1):
            f_eq = self._compute_feq_grad_hermite_terms_by_order(f_eq, order)
            
        print(f'\nNumber of added orthogonal Hermite polynomials : {self.nb_hermite_polynomials}')
        return self._multiply_by_weight_and_simplify(f_eq)
    

