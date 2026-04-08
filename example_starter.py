"""Minimal single-cell example for smoke testing the Birgiolas 2020 MC model."""

from __future__ import annotations

from neuron import h
from prev_ob_models.Birgiolas2020.isolated_cells import MC1

import matplotlib.pyplot as plt


def main() -> None:
    """Run a simple current-clamp pulse and plot the soma voltage response."""
    h.load_file("stdrun.hoc")

    # Use a real single-cell model from the repo instead of the old placeholder template.
    cell = MC1()

    stim = h.IClamp(cell.soma(0.5))
    stim.delay = 50.0
    stim.dur = 100.0
    stim.amp = 0.3  # nA

    soma_voltage = h.Vector().record(cell.soma(0.5)._ref_v)
    time = h.Vector().record(h._ref_t)

    h.finitialize(-65.0)
    h.continuerun(300.0)

    plt.figure(figsize=(10, 4))
    plt.plot(time, soma_voltage)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Voltage (mV)")
    plt.title("MC1 Soma Voltage")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
