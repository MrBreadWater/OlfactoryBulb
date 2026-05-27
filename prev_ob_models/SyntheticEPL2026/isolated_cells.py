"""Literature-constrained synthetic EPL fast-spiking interneuron surrogates.

This module is intentionally isolated from the live slice/network runtime. It
provides a first-pass multicompartment cell family that can be instantiated in
single-cell workflows and discovered through ``prev_ob_models.cell_registry``
without altering the maintained MC/TC/GC simulation path.

Biological target:
- PV+ fast-spiking external plexiform layer interneuron
- strong overlap with the CRH+ axonless EPL population

Design constraints encoded here:
- soma diameter near 9.6 um
- roughly four primary dendrites, matching the reported ~3.5 mean processes
- compact multipolar arbor with strongest branching inside ~30 um of soma
- total dendritic span on the order of ~70 um
- axonless topology

The conductance set reuses the maintained Birgiolas 2020 mechanism library so
the surrogate can live in the same NEURON environment as the rest of the repo.
The parameter values are only a first-pass fast-spiking scaffold, not a fitted
final biological model.
"""

from __future__ import annotations

from math import hypot
from pathlib import Path

from prev_ob_models.utils import IsolatedCell, RunInClassDirectory, load_mechanisms_from_candidates


class _SyntheticEPLCell(IsolatedCell):
    """Shared helper for synthetic EPL interneuron surrogates."""

    sentinel_mechanisms = ("AmpaNmdaSyn", "GabaSyn", "GapJunction", "VecStim", "KainateSyn")
    _instance_counter = 0

    def __init__(self):
        with RunInClassDirectory(type(self)):
            from neuron import h, load_mechanisms

            load_mechanisms_from_candidates(
                load_mechanisms,
                str(Path(__file__).resolve().parents[1] / "Birgiolas2020" / "isolated_cells.py"),
                "Mechanisms",
                sentinel_mechanisms=self.sentinel_mechanisms,
            )

            h.load_file("stdrun.hoc")
            h.celsius = 35.0
            h.cvode_active(1)

            self.h = h
            self.cell = self
            self.synlist = h.List()
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.instance_index = type(self)._instance_counter
            type(self)._instance_counter += 1
            self.instance_name = f"{type(self).__name__}[{self.instance_index}]"

            self._create_sections()
            self._connect_sections()
            self._define_geometry()
            self._create_section_lists()
            self._insert_biophysics()
            self._configure_nseg()

            h.init()

    def _create_sections(self):
        h = self.h
        # Use explicit deterministic names so future slice JSON and section-based
        # bookkeeping do not depend on NEURON's object-repr cell prefixes.
        self.soma = h.Section(name=f"{self.instance_name}.soma")
        self.primary_dendrites = [
            h.Section(name=f"{self.instance_name}.dend_primary_{index}")
            for index in range(4)
        ]
        self.branch_dendrites = [
            h.Section(name=f"{self.instance_name}.dend_branch_{index}")
            for index in range(8)
        ]
        self.dend = list(self.primary_dendrites) + list(self.branch_dendrites)

    def _connect_sections(self):
        for dend in self.primary_dendrites:
            dend.connect(self.soma(0.5), 0.0)
        for index, primary in enumerate(self.primary_dendrites):
            self.branch_dendrites[(2 * index)].connect(primary(1.0), 0.0)
            self.branch_dendrites[(2 * index) + 1].connect(primary(1.0), 0.0)

    def _define_geometry(self):
        self._set_points(
            self.soma,
            [
                (-4.8, 0.0, 0.0, 9.6),
                (0.0, 0.0, 0.0, 9.6),
                (4.8, 0.0, 0.0, 9.6),
            ],
        )

        primary_specs = [
            [(4.8, 0.0, 0.0, 1.4), (20.0, 0.0, 0.0, 1.2)],
            [(0.0, 4.8, 0.0, 1.4), (0.0, 20.0, 0.0, 1.2)],
            [(-4.8, 0.0, 0.0, 1.4), (-20.0, 0.0, 0.0, 1.2)],
            [(0.0, -4.8, 0.0, 1.4), (0.0, -20.0, 0.0, 1.2)],
        ]
        branch_specs = [
            [(20.0, 0.0, 0.0, 1.0), (34.0, 10.0, 2.0, 0.8)],
            [(20.0, 0.0, 0.0, 1.0), (34.0, -10.0, -2.0, 0.8)],
            [(0.0, 20.0, 0.0, 1.0), (10.0, 34.0, 2.0, 0.8)],
            [(0.0, 20.0, 0.0, 1.0), (-10.0, 34.0, -2.0, 0.8)],
            [(-20.0, 0.0, 0.0, 1.0), (-34.0, 10.0, -2.0, 0.8)],
            [(-20.0, 0.0, 0.0, 1.0), (-34.0, -10.0, 2.0, 0.8)],
            [(0.0, -20.0, 0.0, 1.0), (10.0, -34.0, -2.0, 0.8)],
            [(0.0, -20.0, 0.0, 1.0), (-10.0, -34.0, 2.0, 0.8)],
        ]

        for section, points in zip(self.primary_dendrites, primary_specs):
            self._set_points(section, points)
        for section, points in zip(self.branch_dendrites, branch_specs):
            self._set_points(section, points)

    def _create_section_lists(self):
        h = self.h
        self.all = h.SectionList()
        self.somatic = h.SectionList()
        self.dendritic = h.SectionList()

        self.all.append(sec=self.soma)
        self.somatic.append(sec=self.soma)
        for section in self.dend:
            self.all.append(sec=section)
            self.dendritic.append(sec=section)

    def _insert_biophysics(self):
        for section in self.all:
            section.Ra = 120.0
            section.cm = 1.2
            section.insert("pas")
            section.e_pas = -68.0
            section.g_pas = 0.00012
            section.insert("Na")
            section.insert("Kd")
            section.ena = 55.0
            section.ek = -80.0

        self.soma.gbar_Na = 0.32
        self.soma.gbar_Kd = 0.18
        self.soma.insert("KA")
        self.soma.gbar_KA = 0.012
        self.soma.insert("KM")
        self.soma.gbar_KM = 0.0015
        # The Birgiolas Ih mechanism destabilizes this compact surrogate under
        # both CVode and fixed-step protocols. The literature constraints we are
        # using here do not require Ih, so omit it until the fast-spiking
        # scaffold is fitted against dedicated EPL interneuron physiology.

        for section in self.primary_dendrites:
            section.gbar_Na = 0.10
            section.gbar_Kd = 0.06
            section.insert("KA")
            section.gbar_KA = 0.004

        for section in self.branch_dendrites:
            section.gbar_Na = 0.06
            section.gbar_Kd = 0.04
            section.insert("KA")
            section.gbar_KA = 0.002

    def _configure_nseg(self):
        for section in self.all:
            target = max(1, int(round(section.L / 10.0)))
            if target % 2 == 0:
                target += 1
            section.nseg = target

    def _set_points(self, section, points):
        h = self.h
        h.pt3dclear(sec=section)
        for x, y, z, diam in points:
            h.pt3dadd(x, y, z, diam, sec=section)

    def position(self, x, y, z):
        for section in self.all:
            for index in range(int(self.h.n3d(sec=section))):
                self.h.pt3dchange(
                    index,
                    x - self.x + self.h.x3d(index, sec=section),
                    y - self.y + self.h.y3d(index, sec=section),
                    z - self.z + self.h.z3d(index, sec=section),
                    self.h.diam3d(index, sec=section),
                    sec=section,
                )
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def connect2target(self, target, netcon=None):
        netcon = self.h.NetCon(self.soma(0.5)._ref_v, target, sec=self.soma)
        netcon.threshold = 0.0
        return netcon

    @property
    def planar_dendritic_span_um(self):
        coords = []
        for section in self.dend:
            for index in range(int(self.h.n3d(sec=section))):
                coords.append((self.h.x3d(index, sec=section), self.h.y3d(index, sec=section)))
        if not coords:
            return 0.0
        max_span = 0.0
        for x0, y0 in coords:
            for x1, y1 in coords:
                max_span = max(max_span, hypot(x1 - x0, y1 - y0))
        return max_span


class PVCRH_FSI1(_SyntheticEPLCell):
    """Synthetic PV/CRH-overlap fast-spiking EPL interneuron surrogate."""

    literature_constraints = {
        "target_class": "PV+ fast-spiking anaxonic EPL interneuron",
        "overlap_population": "CRH+ EPL interneuron",
        "soma_diameter_um": 9.6,
        "primary_process_count": 4,
        "planar_span_um": 71.0,
        "axon_present": False,
    }


__all__ = ["PVCRH_FSI1"]
