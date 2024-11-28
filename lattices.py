import sympy as sp
import numpy as np
import logging
from utils import kronecker, generate_indices
from constants import *


class Lattice:
    """
    Class to initialize the weights and velocities of the lattice based on D and Q.
    """

    def __init__(self, D=3, Q=19):
        """
        Initialize the lattice with dimension D and number of velocities Q.

        Args:
            D (int): Spatial dimension (1, 2, or 3).
            Q (int): Number of lattice velocities (3, 9, 19, 27).
        """
        self.D = D
        self.Q = Q
        self.Wi = None
        self.Ci = None
        self._initialize_lattice()
        self.print_lattice_info()

    def _initialize_lattice(self):
        """
        Initialize the weights Wi and velocities Ci based on D and Q.
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
                f"The lattice D{self.D}Q{self.Q} is not implemented.")

    def _initialize_D1Q3(self):
        self.Wi = sp.Matrix([2*ONE_3, ONE_6, ONE_6])
        self.Ci = {'x': sp.Matrix([0, 1, -1])}

    def _initialize_D2Q9(self):
        self.Wi = sp.Matrix([4*ONE_9] + [ONE_9]*4 + [ONE_36]*4)
        self.Ci = {
            'x': sp.Matrix([0, 1, 0, -1, 0, 1, -1, -1, 1]),
            'y': sp.Matrix([0, 0, 1, 0, -1, 1, 1, -1, -1]),
            'z': sp.Matrix([0]*9)  # Not used in 2D
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
    #       (0, 0, 0),  # zero
    #       (0, 0, 1), (0, 0, -1),  # z axes
    #       (0, 1, 0), (0, -1, 0),  # y axes
    #       (1, 0, 0), (-1, 0, 0),  # x axes
    #       (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),  # xy planes
    #       (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),  # xz planes
    #       (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)   # yz planes
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

    def print_lattice_info(self):
        """
        Prints the weights Wi and velocities Ci of the lattice based on the logging level.
        """
        # Check if the logging level is at least INFO
        if not logging.getLogger().isEnabledFor(logging.INFO):
            return  # Do nothing if the logging level is higher than INFO
        logging.info(f"Initializing D{self.D}Q{self.Q} lattice.")
        # Convert Wi to a list of strings and join with commas
        weights_list = [str(wi) for wi in self.Wi]
        weights_str = ', '.join(weights_list)
        logging.debug(f"Weights (Wi): [{weights_str}]")
        # Print each component of Ci on a separate line
        for coord in self.Ci:
            # Convert Ci[coord] to a list of strings and join with commas
            ci_list = [str(ci) for ci in self.Ci[coord]]
            ci_str = ', '.join(ci_list)
            logging.debug(f"Velocities (Ci[{coord}]): [{ci_str}]")
