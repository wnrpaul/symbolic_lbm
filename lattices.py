import sympy as sp
import numpy as np
import functools
from utils import kronecker, generate_indices
from constants import *


class Lattice:
    """
    Classe pour initialiser les poids et les vitesses du réseau en fonction de D et Q.
    """

    def __init__(self, D=3, Q=19):
        """
        Initialise le réseau avec la dimension D et le nombre de vitesses Q.

        Args:
            D (int): Dimension spatiale (1, 2 ou 3).
            Q (int): Nombre de vitesses du réseau (3, 9, 19, 27).
        """
        self.D = D
        self.Q = Q
        self.Wi = None
        self.Ci = None
        self._initialize_lattice()

    def _initialize_lattice(self):
        """
        Initialise les poids Wi et les vitesses Ci en fonction de D et Q.
        """
        if self.D == 1 and self.Q == 3:
            self._initialize_D1Q3()
        elif self.D == 2 and self.Q == 9:
            self._initialize_D2Q9()
        elif self.D == 3 and self.Q == 19:
            self._initialize_D3Q19()
        elif self.D == 3 and self.Q == 27:
            self._initialize_D3Q27()
        else:
            raise ValueError(
                f"Le réseau D{self.D}Q{self.Q} n'est pas implémenté.")

    def _initialize_D1Q3(self):
        self.Wi = sp.Matrix([2*ONE_3, ONE_6, ONE_6])
        self.Ci = {'x': sp.Matrix([0, 1, -1])}

    def _initialize_D2Q9(self):
        self.Wi = sp.Matrix([4*ONE_9] + [ONE_9]*4 + [ONE_36]*4)
        self.Ci = {
            'x': sp.Matrix([0, 1, 0, -1, 0, 1, -1, -1, 1]),
            'y': sp.Matrix([0, 0, 1, 0, -1, 1, 1, -1, -1]),
            'z': sp.Matrix([0]*9)  # Non utilisé en 2D
        }

    def _initialize_D3Q19(self):
        self.Wi = sp.Matrix([ONE_3] + [ONE_18]*6 + [ONE_36]*12)
        self.Ci = {
            'x': sp.Matrix([0, 0, 0, 0, 0, 1, -1, 0, 0, 0,
                            0, 1, 1, -1, -1, 1, 1, -1, -1]),
            'y': sp.Matrix([0, 0, 0, 1, -1, 0, 0, 1, 1, -1,
                            -1, 0, 0, 0, 0, 1, -1, 1, -1]),
            'z': sp.Matrix([0, 1, -1, 0, 0, 0, 0, 1, -1, 1,
                            -1, 1, -1, 1, -1, 0, 0, 0, 0])
        }

    # def _initialize_D3Q19_bis(self):
    #   w0 = sp.Rational(1, 3)
    #   w1 = sp.Rational(1, 18)
    #   w2 = sp.Rational(1, 36)
    #   self.Wi = sp.Matrix([w0] + [w1]*6 + [w2]*12)
    #   c = [0, 1, -1]
    #   directions = [
    #       (0, 0, 0),  # zéro
    #       (0, 0, 1), (0, 0, -1),  # axes z
    #       (0, 1, 0), (0, -1, 0),  # axes y
    #       (1, 0, 0), (-1, 0, 0),  # axes x
    #       (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),  # plans xy
    #       (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),  # plans xz
    #       (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)   # plans yz
    #   ]
    #   self.Ci = {
    #       'x': sp.Matrix([dx for dx, dy, dz in directions]),
    #       'y': sp.Matrix([dy for dx, dy, dz in directions]),
    #       'z': sp.Matrix([dz for dx, dy, dz in directions])
    #   }

    def _initialize_D3Q27(self):
        self.Wi = sp.Matrix([8*ONE_27] + [2*ONE_27]*6 +
                            [ONE_54]*12 + [ONE_216]*8)
        self.Ci = {
            'x': sp.Matrix([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1,
                            1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]),
            'y': sp.Matrix([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0,
                            0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1]),
            'z': sp.Matrix([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1,
                            1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1])
        }
