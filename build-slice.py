"""Launch the original Blender/NEURON slice-construction workflow.

This entrypoint is intentionally thin. The actual slice-building logic lives in
``olfactorybulb.slicebuilder`` and in the paired Blender scene/script. The file
is kept because the docs still reference it for manual slice regeneration.
"""

from __future__ import annotations

import argparse
import os
import subprocess

from olfactorybulb.slicebuilder.config import slice_builder_env_overrides_from_cli


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-name", default=None, help="Target slice export directory / Blender slice object name.")
    parser.add_argument("--odors", nargs="+", default=None, help="Odor names to include, or 'all'.")
    parser.add_argument("--max-mcs", type=int, default=None)
    parser.add_argument("--max-tcs", type=int, default=None)
    parser.add_argument("--max-gcs", type=int, default=None)
    parser.add_argument("--enable-epl-interneurons", action="store_true", help="Enable optional EPLI population placement and synapse generation.")
    parser.add_argument("--max-eplis", type=int, default=None, help="Maximum number of optional EPL interneurons to place.")
    parser.add_argument("--epli-particles-object-name", default=None, help="Blender particle object used for EPLI candidate soma positions.")
    parser.add_argument("--epl-interneuron-model", default=None, help="Fully qualified registry key for the EPLI cell model.")
    parser.add_argument("--epl-interneuron-family", default=None, help="Registry family name when selecting by family/role instead of explicit model key.")
    parser.add_argument("--epli-depth-min-fraction", type=float, default=None, help="Lower depth bound within the EPL corridor.")
    parser.add_argument("--epli-depth-max-fraction", type=float, default=None, help="Upper depth bound within the EPL corridor.")
    parser.add_argument("--mc-particles-object-name", default=None)
    parser.add_argument("--tc-particles-object-name", default=None)
    parser.add_argument("--gc-particles-object-name", default=None)
    parser.add_argument("--glom-particles-object-name", default=None)
    parser.add_argument("--glom-layer-object-name", default=None)
    parser.add_argument("--outer-opl-object-name", default=None)
    parser.add_argument("--inner-opl-object-name", default=None)
    parser.add_argument("--blender-executable", default="blender", help="Blender executable to run.")
    parser.add_argument("--blender-file", default="blender-files/ob-gloms-fast.blend", help="Blender scene file to use.")
    parser.add_argument("--print-env", action="store_true", help="Print resolved OB_SLICE_* overrides before launching Blender.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved environment and Blender command without launching NEURON or Blender.")
    return parser


def build_slice(argv: list[str] | None = None):
    """
    To build the model of the slice, most of the work is performed in Blender.

    NEURON is used to instantiate cells, which are exported to Blender, where
    they are positioned and their morphologies modified. These modifications
    are saved to files that NEURON can load later, to run the simulation.

    This file serves as the launcher of NEURON+Blender. It starts by launching
    NEURON with its part of the BlenderNEURON. A few helper methods are added
    to NEURON that can be called from Blender.

    Then, once NEURON is running, in parallel, Blender is started with the BlenderNEURON
    addon. Blender imports cells instantiated in NEURON and uses Blender functions
    to manipulate their morphology.

    Once the cells are positioned, they are saved into files that NEURON can use to
    load the slice model.

    Once the script starts, monitor the console output for progress. After all the cells are
    positioned, connected, and saved, the Blender window will open, showing the model.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env.update(slice_builder_env_overrides_from_cli(args))
    command = [
        args.blender_executable,
        args.blender_file,
        "--python",
        "olfactorybulb/slicebuilder/blender.py",
    ]
    if args.print_env or args.dry_run:
        for key in sorted(name for name in env if name.startswith("OB_SLICE_")):
            print(f"{key}={env[key]}")
    if args.dry_run:
        print("COMMAND:", " ".join(command))
        return 0

    from olfactorybulb.slicebuilder.nrn import SliceBuilderNRN

    # Start NRN and the addon
    sbn = SliceBuilderNRN()

    # Start Blender and build the model
    return subprocess.call(command, env=env)


if __name__ == '__main__':
    build_slice()
