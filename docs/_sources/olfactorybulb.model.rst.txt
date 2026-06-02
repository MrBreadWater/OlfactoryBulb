olfactorybulb.model
=========================================

The main class to build and run the network model. The class constructor builds the model using one of the
specified `parameter classes <olfactorybulb.paramsets.html>`__. If `autorun==True`, the model will be simulated
after building. Otherwise, `run(tstop) <#olfactorybulb.model.OlfactoryBulb.run>`__ is used to run the simulation.

This page predates the maintained OBGPU workflow. The old ``initslice.py``
entrypoint has been removed. For current command-line runs, use
``tools/benchmarks/benchmark_ob.py``; for interactive work, use the maintained
notebook/helper path documented in ``readme.md`` and ``INSTALL.md``.

See the
`LFP Wavelet Analysis.ipynb <https://github.com/JustasB/OlfactoryBulb/blob/master/notebooks/LFP%20Wavelet%20Analysis.ipynb>`_
notebook for examples of how the results are analyzed.

.. automodule:: olfactorybulb.model
    :members:
    :show-inheritance:
