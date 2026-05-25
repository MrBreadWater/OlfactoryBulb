"""OBGPU-friendly wrappers for published Short et al. 2016 cell templates.

These wrappers avoid the legacy GUI-oriented entrypoints and load compiled
mechanisms using the same search strategy as the newer Birgiolas cells.
"""

from prev_ob_models.utils import (
    IsolatedCell,
    RunInClassDirectory,
    load_mechanisms_from_candidates,
)


class _Short2016TemplateCell(IsolatedCell):
    hoc_file = None
    template_name = None
    template_args = ()
    preload_hoc_files = ()
    sentinel_mechanisms = ("kdrmt", "kamt", "nax")
    celsius = 6.3

    def __init__(self):
        if self.hoc_file is None or self.template_name is None:
            raise ValueError("Short2016 template wrappers must set hoc_file and template_name")

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


class PGC(_Short2016TemplateCell):
    """Periglomerular cell template used in the Short et al. 2016 bundle."""

    hoc_file = "PG_def.hoc"
    template_name = "PGcell"
    template_args = (0,)


class ETC(_Short2016TemplateCell):
    """External tufted cell template used in the Short et al. 2016 bundle."""

    hoc_file = "et.hoc"
    template_name = "ET"


__all__ = ["PGC", "ETC"]
