# --------------------------------------------------------
# This is a collection of classes for describing quad
# overlapping fields.
# --------------------------------------------------------
# The classes for the quads with the overlapping fields uses
# the field distribution from the following paper:
#    M.Berz, B. Erdelyn, K.Makino
#    Fringe Field Effects in Small Rings of Large Acceptance
#    Phys. Rev STAB, V3, 124001(2000)
#
# --------------------------------------------------------

import math
import sys
import os
from .quad_field_profile import AbstractQuadFieldSourceFunction
from .enge_function import EngeFunction

from orbit.py_linac.lattice import BaseLinacNode, Drift, Quad
from orbit.py_linac.lattice import AxisFieldRF_Gap
from orbit.py_linac.lattice import AxisField_and_Quad_RF_Gap

# import teapot base functions from wrapper around C++ functions
from orbit.teapot_base import TPB

from orbit.core.orbit_utils import Function


class SimpleQuadFieldFunc(AbstractQuadFieldSourceFunction):
    """
    It is an implementation of the QuadFieldSourceFunction class for
    a simple quad with constant field between (-L/2;+L/2).
    """

    def __init__(self, quad: Quad):
        self.quad = quad

    def getLimitsZ(self) -> tuple[float, float]:
        """
        Returns (+L/2,-L/2) as longitudinal limits of the quad field.
        """
        L = self.quad.getLength()
        return (-L / 2, L / 2)

    def getFuncValue(self, z: float) -> float:
        """
        Returns the quad's normalized field distribution at the distance z from the center
        """
        L = self.quad.getLength()
        if abs(z) <= L / 2:
            return 1.0 / L
        return 0.0

    def getFuncDerivative(self, z: float) -> float:
        """
        Returns the derivative of the getFuncValue(z)
        """
        return 0.0


class PMQ_Trace3D_Function(AbstractQuadFieldSourceFunction):
    """
    The PMQ Function is a represenatation of the field of permanent quad
    from Trace3D documantation (p 77): https://inis.iaea.org/records/7awtr-mr708
    """

    def __init__(self, length_param, rad_in, rad_out, cutoff_level=0.01):
        self.length = length_param
        self.rad_in = rad_in
        self.rad_out = rad_out
        self.cutoff_level = cutoff_level
        self.normalization = 1.0
        self.n_func_points = 500
        z_step = length_param / self.n_func_points
        z_cutoff = self._findCutOff(z_step, cutoff_level)
        self.z_min = -z_cutoff
        self.z_max = +z_cutoff
        self.func = Function()
        self._normalize()

    def _findCutOff(self, step, cutoff_level):
        """Finds the distance from the center where the field is less than cutoff level"""
        init_val = self.getPMQ_FuncValue(0.0)
        z = step
        val = self.getPMQ_FuncValue(z) / init_val
        if val <= cutoff_level:
            return z
        while val > cutoff_level:
            z += step
            val = self.getPMQ_FuncValue(z) / init_val
        z0 = z - step
        z1 = z
        n_inner_points = 100
        step_z = step / n_inner_points
        val0 = self.getPMQ_FuncValue(z0) / init_val
        val1 = self.getPMQ_FuncValue(z1) / init_val
        while abs(z0 - z1) > step_z:
            z_new = (z0 + z1) / 2.0
            val_new = self.getPMQ_FuncValue(z_new) / init_val
            if val_new <= cutoff_level:
                z1 = z_new
                val1 = val_new
            else:
                z0 = z_new
                val0 = val_new
        cutoff_z = (z0 + z1) / 2.0
        return cutoff_z

    def _normalize(self):
        """Normalizes the quad field function to the integral of 1"""
        self.normalization = 1.0
        step = self.z_max / (self.n_func_points - 1)
        sum_int = 0.0
        self.func.clean()
        for ind in range(self.n_func_points):
            z = step * ind
            val = self.getPMQ_FuncValue(z)
            self.func.add(z, val)
            sum_int += val
        sum_int -= (self.getPMQ_FuncValue(0.0) + self.getPMQ_FuncValue(step * (self.n_func_points - 1))) / 2.0
        sum_int *= 2.0 * step
        self.normalization = 1.0 / sum_int

    def getLimitsZ(self):
        """
        Returns (z_min,z_max) tuple as longitudinal limits of the quad field.
        """
        return (self.z_min, self.z_max)

    def pmq_func(self, z):
        """
        This is PMQ function defined at p. 77 of the Trace3D manual.
        """
        r1 = self.rad_in
        r2 = self.rad_out
        v1 = 1.0 / math.sqrt(1.0 + (z / r1) ** 2)
        v2 = 1.0 / math.sqrt(1.0 + (z / r2) ** 2)
        f = 0.5 * (1 - 0.125 * z * (1.0 / r1 + 1.0 / r2) * v1**2 * v2**2 * (v1**2 + v1 * v2 + v1**2 + 4 + 8 / v1 / v2) / (v1 + v2))
        return f

    def getPMQ_FuncValue(self, z):
        """
        Returns the total PMQ quad field distribution
        """
        f = self.pmq_func(z - self.length / 2) - self.pmq_func(z + self.length / 2)
        return f

    def getFuncValue(self, z):
        """Returns the quad's field at the distance z from the center"""
        if abs(z) >= self.func.getMaxX():
            return 0.0
        return self.normalization * self.func.getY(abs(z))

    def getFuncDerivative(self, z):
        """
        Returns the derivative of the getFuncValue(z)
        """
        if abs(z) >= self.func.getMaxX():
            return 0.0
        return math.copysign(self.normalization * self.func.getYP(abs(z)), -z)
