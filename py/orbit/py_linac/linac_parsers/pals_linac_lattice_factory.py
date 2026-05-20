from collections import namedtuple
from typing import Callable

from orbit.aperture import ApertureType
from orbit.py_linac.lattice import (
    AccNode,
    Bend,
    Drift,
    LinacAccLattice,
    MarkerLinacNode,
    Quad,
    RF_Cavity,
    Sequence,
    Solenoid,
)

import pals

PalsElement = pals.Drift | pals.Quadrupole | pals.SBend | pals.RBend | pals.Marker


class PALS_LinacLatticeFactory:
    _elem_registry: dict[str, Callable[[PalsElement], AccNode]] = {}
    _ctx = namedtuple("Context", ["current_pos"])

    def __init__(self):
        self.zero_distance = 1e-5
        self.max_drift_length = 1.0

    def get_linac_acc_lattice(self, names, file_name) -> LinacAccLattice:
        pals_lattice = pals.load(file_name).facility[0]

        lattice = LinacAccLattice(pals_lattice.name)

        branch_length = 0.0
        for branch in lattice.branches:
            sequence = Sequence(branch.name)
            sequence.setLinacAccLattice(lattice)
            sequence.setPosition(branch_length)

            seq_length = 0.0
            for elem in branch.line:
                self._ctx.current_pos = seq_length
                node = self._elem_registry[elem.kind](elem)
                seq_length += node.getLength()
            sequence.setLength(seq_length)
            branch_length += seq_length

        return lattice

    @staticmethod
    def _aperture_type(aperture_params: pals.ApertureParameters) -> ApertureType:
        if aperture_params.shape == "ELLIPTICAL":
            if aperture_params.x_width == aperture_params.y_width:
                return ApertureType.CIRCULAR
            return ApertureType.ELLIPTICAL

        if aperture_params.shape == "RECTANGULAR":
            return ApertureType.RECTANGULAR

        return ApertureType.CIRCULAR

    def _nparts(self, node_length: float) -> int:
        return 2 * int(node_length / self.max_drift_length + 1.5 - 1.0e-12)

    def _register(self, node_type: str):
        def decorator(method):
            self._elem_registry[node_type] = method
            return method

        return decorator

    @_register("Drift")
    def build_drift(self, elem: PalsElement) -> Drift:
        node = Drift(elem.name)
        node.setLength(elem.length)
        node.setParam("pos", self._ctx.current_pos)
        return node

    @_register("Quadrupole")
    def build_quad(self, elem: PalsElement) -> Quad:
        # TODO: implement multipoles; what do we do with radIn/radOut?
        node = Quad(elem.name)
        node.setLength(elem.length)
        node.setParam("pos", self._ctx.current_pos)
        node.setField(elem.Bn1)
        half_length = 0.5 * node.getLength()
        if half_length > self.max_drift_length:
            node.setnParts(self._nparts(half_length))

        if elem.MagneticMultipoleP.Bn0 != 0.0:
            pass
        if elem.MagneticMultipoleP.Bs0 != 0.0:
            pass

        if elem.ApertureP.aperture_active:
            node.setParam("aprt_type", self._aperture_type(elem.ApertureP))
            node.setParam("aperture", elem.ApertureP.x_width)

        return node

    @_register("Marker")
    def build_marker(self, elem: PalsElement) -> MarkerLinacNode:
        node = MarkerLinacNode(elem.name)
        node.setParam("pos", self._ctx.current_pos)
        return node

    @_register("RFCavity")
    def build_rfcavity(self, elem: PalsElement) -> RF_Cavity:
        raise NotImplementedError

    @_register("SBend")
    def build_sbend(self, elem: PalsElement) -> Bend:
        # TODO: implement apertures; multipoles; what do we do with radIn/radOut?
        node = Bend(elem.name)
        node.setLength(elem.length)
        node.setParam("pos", self._ctx.current_pos)
        node.setParam("ea1", elem.BendP.e1)
        node.setParam("ea2", elem.BendP.e2)
        node.setParam("theta", elem.BendP.rho_ref)
        if node.getLength() > self.max_drift_length:
            node.setnParts(self._nparts(node.getLength()))
        return node

    @_register("Solenoid")
    def build_solenoid(self, elem: PalsElement) -> Solenoid:
        node = Solenoid(elem.name)
        node.setLength(elem.length)
        node.setParam("pos", self._ctx.current_pos)
        node.setParam("B", elem.SolenoidP.Bsol)
        return node
