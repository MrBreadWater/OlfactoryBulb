"""OBGPU-friendly wrappers for published Li and Cleland 2013 cell templates.

These wrappers load the direct HOC templates instead of the legacy *Stim.hoc
entrypoints so they can be used cleanly in notebooks and scripted workflows.
"""

from prev_ob_models.utils import (
    IsolatedCell,
    RunInClassDirectory,
    load_mechanisms_from_candidates,
)


class _LiCleland2013TemplateCell(IsolatedCell):
    hoc_file = None
    template_name = None
    template_args = (0,)
    preload_hoc_files = ()
    sentinel_mechanisms = ("kdrmt", "kamt", "nax")
    celsius = 35.0

    def __init__(self):
        if self.hoc_file is None or self.template_name is None:
            raise ValueError("LiCleland2013 template wrappers must set hoc_file and template_name")

        with RunInClassDirectory(type(self)):
            from neuron import h, load_mechanisms

            load_mechanisms_from_candidates(
                load_mechanisms,
                __file__,
                mechanism_dir_name=".",
                sentinel_mechanisms=self.sentinel_mechanisms,
            )

            h.load_file("stdrun.hoc")
            for hoc_file in self.preload_hoc_files:
                h.load_file(hoc_file)
            h.load_file(self.hoc_file)

            template = getattr(h, self.template_name)
            self.h = h
            self.cell = template(*self.template_args)
            self.soma = self.cell.soma

            h.celsius = self.celsius
            h.cvode_active(1)
            h.init()


class PGC(_LiCleland2013TemplateCell):
    """Periglomerular cell from Li and Cleland (2013)."""

    hoc_file = "PG_def.hoc"
    template_name = "PGcell"


class MC(_LiCleland2013TemplateCell):
    """Mitral cell from Li and Cleland (2013)."""

    hoc_file = "MC_def.hoc"
    template_name = "Mitral"
    preload_hoc_files = ("tabchannels.hoc",)


class GC(_LiCleland2013TemplateCell):
    """Granule cell from Li and Cleland (2013)."""

    hoc_file = "GC_def.hoc"
    template_name = "Granule"


__all__ = ["PGC", "MC", "GC"]
