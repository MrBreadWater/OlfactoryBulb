"""Build and run the olfactory bulb network model.

This module is the main simulation entrypoint used by the current OBGPU
workflow. It combines:

- cell instantiation from the Birgiolas 2020 templates
- JSON-driven slice connectivity
- odor/input scheduling
- optional CoreNEURON/GPU execution
- output saving for soma voltages, LFP, and notebook analysis helpers

The file is historically dense because it grew around both the original NEURON
workflow and the newer CoreNEURON/OBGPU parity work.
"""

try:
    import cPickle # Python 2
except:
    import pickle as cPickle # Python 3

import os
import numpy as np
import json
from types import SimpleNamespace
from prev_ob_models.Birgiolas2020.isolated_cells import *
from blenderneuron.nrn.neuronnode import NeuronNode
from olfactorybulb.database import Odor, OdorGlom, CellModel, database
from math import pow
from LFPsimpy import LfpElectrode, SectionLfpLineMethod
import sys
from heapq import *
from matplotlib import pyplot as plt
from hashlib import sha1
from random import random, seed

from olfactorybulb.output_paths import get_results_dir, write_run_info
from olfactorybulb.paramsets.base import *
from olfactorybulb.paramsets.case_studies import *
from olfactorybulb.paramsets.sensitivity import *
from olfactorybulb.inputs import InputSpec
from olfactorybulb.result_artifacts import (
    DEFAULT_SOMA_TRACE_DTYPE,
    DEFAULT_SOMA_TRACE_FORMAT,
    DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
    DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
    DEFAULT_SOMA_SPIKE_THRESHOLD_MV,
    SOMA_TRACE_FILENAME_NPZ,
    SOMA_TRACE_FILENAME_PKL,
    save_soma_spike_artifact,
    save_soma_trace_artifact,
    save_voltage_summary_artifact,
)

CELL_MODEL_FACTORIES = {
    name: value
    for name, value in globals().items()
    if isinstance(value, type)
}


def should_force_gid_synapses(params, nranks):
    """Return whether reciprocal synapses should route through segment gids.

    Plain single-rank NEURON runs historically used direct voltage-source
    NetCons. Forcing gid-based event routing there changes local dynamics and
    can destabilize the legacy CPU path. Keep gid routing for multi-rank or
    CoreNEURON execution, and allow an explicit param override either way.
    """

    explicit = getattr(params, "force_gid_synapses", None)
    if explicit is not None:
        return bool(explicit)

    coreneuron_cfg = getattr(params, "coreneuron", None)
    coreneuron_enabled = bool(
        coreneuron_cfg is not None and getattr(coreneuron_cfg, "enable", False)
    )
    return int(nranks) > 1 or coreneuron_enabled


class OBNeuronNode(NeuronNode):
    """Use segment gids for synaptic event sources on every rank.

    CoreNEURON rejects MPI runs that mix direct voltage-source NetCons for
    local connections with gid-based connections for remote ones. Routing all
    BlenderNEURON-generated synapses through segment gids keeps the event path
    uniform across ranks and matches the existing cross-rank behavior.
    """

    def create_netcon_syn(
        self,
        syn_class_name,
        syn_sec,
        syn_sec_x,
        syn_params,
        source_sec,
        source_x,
        threshold,
        delay,
        weight,
        source_on_rank,
        syn_on_rank,
        source_gid,
    ):
        from neuron import h

        netcon, syn = None, None

        if syn_on_rank:
            syn_class = getattr(h, syn_class_name)
            syn = syn_class(syn_sec(syn_sec_x))

            if syn_params != "":
                syn_params = eval(syn_params)
                for key, value in syn_params.items():
                    setattr(syn, key, value)

        use_gid_netcons = (
            self.parallel_ctx is not None
            and self.mpimap is not None
            and getattr(self, "force_gid_synapses", True)
        )

        if self.parallel_ctx is not None and self.mpimap is not None and source_on_rank:
            if self.parallel_ctx.gid_exists(source_gid) == 0:
                self.assign_gid_to_source_seg(source_sec, source_x, threshold, source_gid)

        if use_gid_netcons and (source_on_rank or syn_on_rank):

            if syn_on_rank:
                netcon = self.parallel_ctx.gid_connect(source_gid, syn)
                netcon.delay = delay
                netcon.weight[0] = weight

        elif syn_on_rank and source_on_rank:
            netcon = h.NetCon(
                source_sec(source_x)._ref_v,
                syn,
                threshold,
                delay,
                weight,
                sec=source_sec,
            )

        elif source_on_rank or syn_on_rank:
            if source_on_rank:
                if self.parallel_ctx.gid_exists(source_gid) == 0:
                    self.assign_gid_to_source_seg(source_sec, source_x, threshold, source_gid)
            else:
                netcon = self.parallel_ctx.gid_connect(source_gid, syn)
                netcon.delay = delay
                netcon.weight[0] = weight

        return netcon, syn

    def remember_cell_source_gid(self, section_name, gid):
        if not hasattr(self, "cell_source_gids"):
            self.cell_source_gids = {}

        cell_name = section_name.split(".", 1)[0]
        self.cell_source_gids.setdefault(cell_name, int(gid))

    def create_synapses(self, syn_set):
        synapses = super().create_synapses(syn_set)

        for entry in syn_set["entries"]:
            rank_source_section = self.rank_section_name(entry["source_section"])
            if rank_source_section is not None:
                source_gid = self.segment_gid(
                    entry["source_section"], entry["source_seg_i"], entry["create_spine"]
                )
                self.remember_cell_source_gid(rank_source_section, source_gid)

            if entry["is_reciprocal"]:
                rank_dest_section = self.rank_section_name(entry["dest_section"])
                if rank_dest_section is not None:
                    dest_gid = self.segment_gid(entry["dest_section"], entry["dest_seg_i"], False)
                    self.remember_cell_source_gid(rank_dest_section, dest_gid)

        return synapses


class ParallelSafeLfpElectrode(LfpElectrode):
    """LFPsimpy wrapper that avoids per-sample Python gathers under MPI."""

    def compute(self):
        if self.h.t == 0:
            return 0

        result = sum(sec_lfp.compute() for sec_lfp in self.section_lfps.values())

        if self.parallel_ctx.id() == 0:
            self.nrn_value_tracker.value = result

        return result


class OlfactoryBulb:
    """
    The main class used to build and simulate the olfactory bulb network model.
    """

    def __init__(self, params="ParameterSetBase", autorun=True):
        """
        :param params: The name of the class defined in olfactorybulb.paramsets that defines the network parameters
        :param autorun: When true, after the network model is built, starts the simulation
        """

        if type(params) == str:
            params = eval(params)()

        self.params = params

        self.rnd_seed = params.rnd_seed

        self.slice_dir = os.path.abspath(os.path.join(params.slice_dir, params.slice_name))
        self.cells = {}
        self.inputs = []
        self.kar_inputs = []
        self.gc_kar_synapses = []

        self.gj_source_gids = set()
        self.gjs = []
        self._stable_hash_cache = {}
        self._segment_cache = {}
        self._rank_section_name_cache = {}
        self._model_inputsegs_cache = None
        self._model_nseg_count_cache = {}
        self._odor_glom_intensities_cache = {}
        self._glom_input_seg_cache = None
        self._gap_junction_seg_cache = {}
        self._native_lfp_objects = []
        self._next_lfp_report_gid = 1500000000
        self._native_lfp_prepared = False
        self._native_lfp_report_path = None
        self._native_lfp_report_conf_path = None
        self._native_lfp_sim_conf_path = None
        self._native_lfp_cell_gids = {}
        self._native_lfp_gid_source = {}
        self._native_lfp_mappings_registered = False
        self._status_is_tty = sys.stdout.isatty()
        self._status_mode = os.environ.get("OBGPU_STATUS_MODE", "auto").strip().lower()
        self._last_status_percent = None
        self._last_status_ms = None

        from neuron import h, load_mechanisms
        self.h = h
        self.pc = h.ParallelContext()
        self.mpimap = {}
        self.nranks = int(self.pc.nhost())
        self.mpirank = self.pc.id()
        self._next_lfp_report_gid = 1500000000 + (self.mpirank * 1000000)
        self._next_input_source_gid = 2000000000 + (self.mpirank * 1000000)

        # Just use the BlenderNEURON package functions (e.g. no server/client)
        self.bn_server = OBNeuronNode(server_end='Package')
        self.bn_server.force_gid_synapses = should_force_gid_synapses(params, self.nranks)
        self.bn_server.cell_source_gids = {}

        # Keep track of rank complexities with a min-heap
        self.rank_complexities = [(0, r) for r in range(self.nranks)]

        coreneuron_cfg = getattr(params, "coreneuron", None)
        self._use_dense_soma_recording = bool(
            coreneuron_cfg is not None and getattr(coreneuron_cfg, "enable", False)
        )
        self._actual_dt = float(params.sim_dt)

        self.t_vec = h.Vector()
        self.t_vec.record(h._ref_t, params.recording_period)
        self.v_vectors = {}
        self.input_vectors = []
        self.gc_output_event_vectors = []

        legacy_group_loading = bool(getattr(self.params, "legacy_group_loading", False))
        if legacy_group_loading:
            for cell_type in ['MC', 'GC', 'TC']:
                group_dict = self.load_cells(cell_type)
                self.finish_loading_cells([group_dict])
        else:
            cell_groups = []
            for cell_type in ['MC', 'GC', 'TC']:
                cell_groups.append(self.load_cells(cell_type))
            self.finish_loading_cells(cell_groups)

        if self.mpirank == 0:
            complexities = np.array([c[0] for c in self.rank_complexities])
            min = np.min(complexities)
            max = np.max(complexities)
            mean = np.mean(complexities)

            print('Rank Complexity min: %s, mean: %s, max: %s' % (min, mean, max))

        self.apply_gc_ka_gbar_scale()

        if getattr(self.params, "enable_reciprocal_synapses", True):
            for synapse_set in ['GCs__MCs', 'GCs__TCs']:
                self.load_synapse_set(synapse_set)
            if getattr(self.params, "enable_gc_kar", False) and float(getattr(self.params, "kar_gc_gmax", 0.0)) > 0:
                for synapse_set in ['GCs__MCs', 'GCs__TCs']:
                    self.add_gc_kar_synapse_set(synapse_set)
            if getattr(self.params, "record_gc_output_events", False):
                self.record_gc_output_events()

        # Load glom->cell links
        self.load_glom_cells()

        # Create gap junctions between MC and TC tufts
        for cell_type, g_gap in params.gap_juction_gmax.items():
            self.add_gap_junctions(cell_type, g_gap)

        # Set synapse parameters
        for syn_mech, syn_values in params.synapse_properties.items():
            if hasattr(h, syn_mech):
                for syn_attrib, attrib_value in syn_values.items():
                    for synapse in getattr(h, syn_mech):
                        setattr(synapse, syn_attrib, attrib_value)

        _has_odors = bool(getattr(params, "input_odors", {}))
        _has_stimuli = bool(getattr(params, "input_stimuli", {}))
        assert not (_has_odors and _has_stimuli), (
            "input_odors and input_stimuli cannot both be non-empty. "
            "Use input_odors for odor-DB-driven inputs or input_stimuli for "
            "custom InputSpec-driven inputs, but not both at once."
        )

        # Add glomerular inputs
        for time, odor_info in params.input_odors.items():
            self.add_inputs(odor=odor_info["name"], t=time, rel_conc=odor_info["rel_conc"])

        # Add custom InputSpec-based stimuli
        for time, entry in getattr(params, "input_stimuli", {}).items():
            if isinstance(entry, InputSpec):
                spec, intensity, cell_types = entry, 1.0, None
            else:
                spec = entry["input"]
                intensity = float(entry.get("intensity", 1.0))
                cell_types = entry.get("cell_types", None)
            self.add_stimulus_inputs(t=float(time), input_spec=spec,
                                     intensity=intensity, cell_types=cell_types)

        # LFP electrode creation is deferred until run() so modern NEURON can
        # set up ParallelContext transfer state before any implicit h.init().
        self.electrode = None
        self._electrode_kwargs = {
            "x": params.lfp_electrode_location[0],
            "y": params.lfp_electrode_location[1],
            "z": params.lfp_electrode_location[2],
            "sampling_period": params.recording_period,
            "method": "Line",
        }

        if self.use_corenrn_native_lfp():
            self.register_corenrn_native_lfp_mappings()

        self.setup_status_reporter()

        for cell_type in params.record_from_somas:
            self.record_from_somas(cell_type)

        if (
            self.mpirank == 0
            and self.nranks == 1
            and getattr(self.params, "enable_lfp", True)
            and not self.use_corenrn_native_lfp()
        ):
            from neuron import gui
            # h.load_file('1x1x1-testbed.ses')

            h.newPlotI()
            [g for g in h.Graph][-1].addvar('LfpElectrode[0].value')

        if autorun:
            self.run(params.tstop)

            if self.mpirank == 0:
                self.ensure_results_dir()

            self.save_recorded_vectors()

            self.get_lfp()

            # Cleanup on MPI
            if self.nranks > 1:
                database.close()
                self.h.quit()


    def stim_glom_segments(self, time, input_segs, intensity, input_spec=None):
        """
        Adds input synapses onto glomerular tufts at specified start time and intensity.

        When ``input_spec`` is None the existing Gaussian spike-train behavior is
        used (backward compatible).  When an InputSpec is provided its
        ``generate_spike_times`` method is called instead; per-segment independent
        randomization is preserved through the same seed scheme.

        :param time: the inhalation onset time in ms
        :param input_segs: list of (rank_seg_name, seg_gid, single_rank_seg_name) tuples
        :param intensity: 0-1 scaling factor
        :param input_spec: optional InputSpec; None uses the Gaussian default
        """

        h = self.h

        inhale_duration = self.params.inhale_duration
        max_firing_rate = self.params.max_firing_rate

        # Only used for the Gaussian fallback path
        spike_count = int(round(max_firing_rate * intensity * (inhale_duration / 1000.0)))

        for seg_name, seg_gid, single_rank_seg_name in input_segs:
            # Per-segment independent RNG seeded identically to the existing scheme
            seed_source = "%s|%s|%s|%s" % (self.rnd_seed, time, single_rank_seg_name, intensity)
            rng = np.random.RandomState(self.stable_hash(seed_source))

            if input_spec is not None:
                spike_times = input_spec.generate_spike_times(time, rng, intensity)
            else:
                spike_times = self.get_gaussian_spike_train(spike_count, time, inhale_duration, rng=rng)

            # Create synapse point process
            seg = self.resolve_segment(seg_name)
            syn = h.Exp2Syn(seg)
            syn.tau1 = self.params.input_syn_tau1
            syn.tau2 = self.params.input_syn_tau2

            if "MC" in seg_name:  # MCs
                delay = self.params.mc_input_delay
                weight = self.params.mc_input_weight

            else:  # "TC"
                delay = self.params.tc_input_delay
                weight = self.params.tc_input_weight

            kar_syn = self.create_osn_kar_synapse(seg)
            kar_weight = float(weight) * float(getattr(self.params, "kar_osn_weight_scale", 1.0))

            event_times = [float(t_event) for t_event in spike_times + delay]
            input_event_strategy = getattr(self.params, "input_event_strategy", None)
            if input_event_strategy is None:
                coreneuron_cfg = getattr(self.params, "coreneuron", None)
                if coreneuron_cfg is not None and getattr(coreneuron_cfg, "enable", False):
                    input_event_strategy = "scheduled"
                else:
                    input_event_strategy = "vecstim"

            if input_event_strategy == "vecstim":
                ns = h.VecStim()
                event_vec = h.Vector(event_times)
                ns.play(event_vec)

                netcon = h.NetCon(
                    ns,
                    syn,
                    0,
                    0,
                    weight,
                )
                kar_netcon = None
                if kar_syn is not None:
                    kar_netcon = h.NetCon(
                        ns,
                        kar_syn,
                        0,
                        0,
                        kar_weight,
                    )

                input_vec = h.Vector()
                netcon.record(input_vec)
                self.input_vectors.append((single_rank_seg_name, input_vec))
                self.inputs.append((syn, ns, netcon, event_vec, kar_syn, kar_netcon))
            elif input_event_strategy == "patternstim":
                input_delay = 2.0 * float(self.params.sim_dt)
                source_gid = self._next_input_source_gid
                self._next_input_source_gid += 1

                netcon = self.pc.gid_connect(source_gid, syn)
                netcon.delay = input_delay
                netcon.weight[0] = weight
                kar_netcon = None
                if kar_syn is not None:
                    kar_netcon = self.pc.gid_connect(source_gid, kar_syn)
                    kar_netcon.delay = input_delay
                    kar_netcon.weight[0] = kar_weight

                scheduled_event_times = [float(t_event - input_delay) for t_event in event_times]
                pattern_times = h.Vector(scheduled_event_times)
                pattern_gids = h.Vector([float(source_gid)] * len(scheduled_event_times))
                pattern_stim = h.PatternStim()
                pattern_stim.fake_output = 1
                pattern_stim.play(pattern_times, pattern_gids)

                input_vec = h.Vector(event_times)
                self.input_vectors.append((single_rank_seg_name, input_vec))
                self.inputs.append((syn, netcon, pattern_stim, pattern_times, pattern_gids, kar_syn, kar_netcon))
            else:
                # Use a nil-source NetCon and schedule the precomputed events during
                # finitialize. This avoids the custom VecStim artificial cell while
                # preserving the externally generated spike times.
                # For nil-source NetCons, nc.event(t) schedules an event for absolute
                # delivery time t; nc.delay is not added to that delivery time. Keep
                # delay > dt for CoreNEURON's minimum-delay bookkeeping, but queue
                # events at their intended delivery times.
                input_delay = 2.0 * float(self.params.sim_dt)
                netcon = h.NetCon(None, syn)
                netcon.delay = input_delay
                netcon.weight[0] = weight
                event_netcons = [netcon]
                kar_netcon = None
                if kar_syn is not None:
                    kar_netcon = h.NetCon(None, kar_syn)
                    kar_netcon.delay = input_delay
                    kar_netcon.weight[0] = kar_weight
                    event_netcons.append(kar_netcon)

                input_vec = h.Vector()
                self.input_vectors.append((single_rank_seg_name, input_vec))

                def schedule_events(
                    ncs=tuple(event_netcons),
                    scheduled_times=tuple(event_times),
                    input_record=input_vec,
                    h_ref=self.h,
                ):
                    input_record.resize(0)
                    tstop = float(h_ref.tstop)
                    for event_time in scheduled_times:
                        if event_time <= tstop + 1e-9:
                            for nc in ncs:
                                nc.event(event_time)
                            input_record.append(event_time)

                fih = h.FInitializeHandler(1, schedule_events)
                self.inputs.append((syn, netcon, fih, kar_syn, kar_netcon))

    def stable_hash(self, source, digits=9):
        """
        Creates a hash code of digits long that is stable across different machines.

        :param source: The string to hash, in this case a section name
        :param digits: The number of digits to keep of the hash
        :return: The hash code as an integer
        """

        key = (source, digits)
        cached = self._stable_hash_cache.get(key)
        if cached is None:
            cached = int(sha1(source.encode()).hexdigest(), 16) % (10 ** digits)
            self._stable_hash_cache[key] = cached
        return cached

    def resolve_segment(self, seg_name):
        normalized_name = seg_name.replace('(1)', '(.999)')
        seg = self._segment_cache.get(normalized_name)
        if seg is None:
            seg = eval(normalized_name, {"h": self.h})
            self._segment_cache[normalized_name] = seg
        return seg

    def require_mechanism(self, mechanism_name):
        if not hasattr(self.h, mechanism_name):
            raise RuntimeError(
                "%s is not available in the loaded NEURON mechanisms. "
                "Run nrnivmodl after adding or updating mechanism .mod files."
                % mechanism_name
            )

    def configure_kar_synapse(self, syn, gmax):
        syn.gmax = float(gmax)
        for mod_name in ("tau1", "tau2", "tau3", "amp1", "amp2", "amp3"):
            param_name = "kar_%s" % mod_name
            if hasattr(syn, mod_name) and hasattr(self.params, param_name):
                setattr(syn, mod_name, float(getattr(self.params, param_name)))
        syn.kd = float(getattr(self.params, "kar_kd", 0.0))
        syn.e = float(getattr(self.params, "kar_e", 0.0))
        syn.block = float(getattr(self.params, "kar_block", 1.0))

    def create_osn_kar_synapse(self, seg):
        if not bool(getattr(self.params, "enable_osn_kar", True)):
            return None

        gmax = float(getattr(self.params, "kar_mt_gmax", 0.0))
        if gmax <= 0:
            return None

        self.require_mechanism("KainateSyn")
        syn = self.h.KainateSyn(seg)
        self.configure_kar_synapse(syn, gmax)
        self.kar_inputs.append(syn)
        return syn

    def apply_gc_ka_gbar_scale(self):
        scale = float(getattr(self.params, "gc_ka_gbar_scale", 1.0))
        if scale == 1.0:
            return

        h = self.h
        for cell_model in self.cells.get("GC", []):
            for sec in self.get_cell_sections(cell_model):
                try:
                    has_ka = bool(h.ismembrane("KA", sec=sec))
                except Exception:
                    has_ka = hasattr(sec, "gbar_KA")
                if not has_ka or not hasattr(sec, "gbar_KA"):
                    continue
                sec.gbar_KA = float(sec.gbar_KA) * scale

    def rank_section_name(self, cell_name):
        cached = self._rank_section_name_cache.get(cell_name)
        if cached is None:
            cached = self.bn_server.rank_section_name(cell_name)
            self._rank_section_name_cache[cell_name] = cached
        return cached

    def use_corenrn_native_lfp(self):
        coreneuron_cfg = getattr(self.params, "coreneuron", None)
        return bool(
            coreneuron_cfg is not None
            and getattr(coreneuron_cfg, "enable", False)
            and getattr(self.params, "enable_lfp", True)
        )

    def configure_corenrn_runtime(self):
        coreneuron_cfg = getattr(self.params, "coreneuron", None)
        if coreneuron_cfg is None or not getattr(coreneuron_cfg, "enable", False):
            return None

        from neuron import coreneuron

        coreneuron.enable = True
        coreneuron.gpu = bool(getattr(coreneuron_cfg, "gpu", False))
        coreneuron.file_mode = bool(getattr(coreneuron_cfg, "file_mode", False))
        coreneuron.verbose = int(getattr(coreneuron_cfg, "verbose", 0))

        cell_permute = getattr(coreneuron_cfg, "cell_permute", None)
        if cell_permute is None:
            cell_permute = 2 if coreneuron.gpu else 0
        coreneuron.cell_permute = int(cell_permute)

        coreneuron.sim_config = ""
        return coreneuron

    def get_results_dir(self):
        if not hasattr(self, "results_dir"):
            self.results_dir = str(get_results_dir(self.params.name))
        return self.results_dir

    def ensure_results_dir(self):
        results_dir = self.get_results_dir()
        if self.mpirank == 0:
            run_info = {
                "paramset": self.params.name,
                "timestamp": os.environ.get("OB_RUN_TIMESTAMP"),
                "result_label": os.path.basename(results_dir),
                "results_dir": results_dir,
                "nranks": int(self.nranks),
                "recording_period": float(self.params.recording_period),
                "sim_dt": float(self.params.sim_dt),
                "legacy_parallel_dt": bool(getattr(self.params, "legacy_parallel_dt", True)),
                "runtime_mode": getattr(self.params, "runtime_mode", "scientific"),
                "coreneuron": {
                    "enable": bool(getattr(getattr(self.params, "coreneuron", None), "enable", False)),
                    "gpu": bool(getattr(getattr(self.params, "coreneuron", None), "gpu", False)),
                    "file_mode": bool(getattr(getattr(self.params, "coreneuron", None), "file_mode", False)),
                    "verbose": int(getattr(getattr(self.params, "coreneuron", None), "verbose", 0)),
                    "cell_permute": getattr(getattr(self.params, "coreneuron", None), "cell_permute", None),
                },
            }
            write_run_info(results_dir, run_info)
        return results_dir

    def iter_cell_models(self):
        for cells in self.cells.values():
            for cell_model in cells:
                yield cell_model

    def get_cell_sections(self, cell_model):
        sec_list = self.h.SectionList()
        sec_list.wholetree(sec=cell_model.soma)
        return list(sec_list)

    def get_cell_name(self, cell_model):
        return cell_model.soma.name().split(".", 1)[0]

    def remember_cell_source_gid(self, section_name, gid):
        cell_name = section_name.split(".", 1)[0]
        self._native_lfp_gid_source.setdefault(cell_name, int(gid))

    def get_cell_report_gid(self, cell_model):
        gid = self._native_lfp_cell_gids.get(id(cell_model))
        if gid is None:
            gid = int(self._next_lfp_report_gid)
            self._next_lfp_report_gid += 1
            self._native_lfp_cell_gids[id(cell_model)] = gid
        return gid

    def register_corenrn_lfp_mapping_for_cell(self, gid, cell_model):
        h = self.h
        nc = None
        if self.pc.gid_exists(gid) == 0:
            nc = h.NetCon(cell_model.soma(0.5)._ref_v, None, sec=cell_model.soma)
            nc.threshold = 1e9
            self.pc.set_gid2node(gid, self.mpirank)
            self.pc.cell(gid, nc)

        electrode_geometry = SimpleNamespace(
            h=h,
            elec_x=self._electrode_kwargs["x"],
            elec_y=self._electrode_kwargs["y"],
            elec_z=self._electrode_kwargs["z"],
        )

        sec_ids = []
        seg_ids = []
        lfp_factors = []

        for sec_index, sec in enumerate(self.get_cell_sections(cell_model)):
            transfer_resistance = float(SectionLfpLineMethod(electrode_geometry, sec).transfer_resistance)
            for seg in sec:
                sec_ids.append(float(sec_index))
                seg_ids.append(float(seg.node_index()))
                lfp_factors.append(transfer_resistance)

        if not seg_ids:
            return

        sec_vec = h.Vector(sec_ids)
        seg_vec = h.Vector(seg_ids)
        lfp_vec = h.Vector(lfp_factors)
        self.pc.nrnbbcore_register_mapping(gid, "All", sec_vec, seg_vec, lfp_vec, 1)
        self._native_lfp_objects.append((nc, sec_vec, seg_vec, lfp_vec))

    def register_corenrn_native_lfp_mappings(self):
        if self._native_lfp_mappings_registered or not self.use_corenrn_native_lfp():
            return

        for cell_model in self.iter_cell_models():
            gid = self.get_cell_report_gid(cell_model)
            if gid is None:
                gid = self._next_lfp_report_gid
                self._next_lfp_report_gid += 1
            self.register_corenrn_lfp_mapping_for_cell(gid, cell_model)

        self._native_lfp_mappings_registered = True

    def write_corenrn_lfp_report_config(self, gids):
        results_dir = self.get_results_dir()
        self._native_lfp_report_path = os.path.join(results_dir, "lfp_native.tsv")
        self._native_lfp_report_conf_path = os.path.join(results_dir, "lfp_native.report.conf")
        self._native_lfp_sim_conf_path = os.path.join(results_dir, "lfp_native.sim.conf")

        with open(self._native_lfp_report_conf_path, "wb") as f:
            metadata = (
                f"1\n"
                f"lfp_native.tsv All lfp v nV SONATA All Center "
                f"{float(self.params.recording_period):g} 0.0 {float(self.h.tstop):g} {len(gids)} 8 None\n"
            )
            f.write(metadata.encode("ascii"))
            if gids:
                gid_bytes = np.asarray(gids, dtype=np.int32).tobytes()
                f.write(gid_bytes)
            f.write(b"\n")
            f.write(b"1\n")
            f.write(b"All 0\n")
            f.write(b"out.h5\n")

        with open(self._native_lfp_sim_conf_path, "w") as f:
            f.write(f"outpath='{results_dir}'\n")
            f.write(f"report-conf='{self._native_lfp_report_conf_path}'\n")

    def prepare_corenrn_native_lfp(self):
        if self._native_lfp_prepared or not self.use_corenrn_native_lfp():
            return

        from neuron import coreneuron

        results_dir = self.get_results_dir()
        self._native_lfp_report_path = os.path.join(results_dir, "lfp_native.tsv")
        self._native_lfp_report_conf_path = os.path.join(results_dir, "lfp_native.report.conf")
        self._native_lfp_sim_conf_path = os.path.join(results_dir, "lfp_native.sim.conf")

        if self.mpirank == 0 and not os.path.exists(results_dir):
            self.ensure_results_dir()
        self.pc.barrier()

        self.register_corenrn_native_lfp_mappings()
        local_gids = []
        for cell_model in self.iter_cell_models():
            gid = self.get_cell_report_gid(cell_model)
            if gid is None:
                gid = self._next_lfp_report_gid
                self._next_lfp_report_gid += 1
            local_gids.append(gid)

        all_gid_lists = self.pc.py_gather(local_gids, 0)
        if self.mpirank == 0:
            flat_gids = [gid for rank_gids in all_gid_lists for gid in rank_gids]
            self.write_corenrn_lfp_report_config(flat_gids)

        self.pc.barrier()
        coreneuron.sim_config = self._native_lfp_sim_conf_path
        self._native_lfp_prepared = True

    def run(self, tstop):
        """
        Runs the NEURON simulation until the specified stop time

        :param tstop: Simulation stop time
        """

        if self.mpirank == 0:
            print('Starting simulation...')

        h = self.h
        h.tstop = tstop
        if self.mpirank == 0:
            self.write_progress_status(float(h.t), float(h.tstop))

        coreneuron_cfg = getattr(self.params, "coreneuron", None)
        coreneuron_enabled = bool(coreneuron_cfg is not None and getattr(coreneuron_cfg, "enable", False))
        native_corenrn_lfp = self.use_corenrn_native_lfp()
        if coreneuron_enabled:
            self.configure_corenrn_runtime()

        # Gap junctions are connected through ParallelContext transfer variables.
        # Even on a single rank, those models need the psolve path instead of h.run().
        uses_parallel_transfer = self.nranks > 1 or len(self.gjs) > 0 or coreneuron_enabled

        if not uses_parallel_transfer:
            if getattr(self.params, "enable_lfp", True) and not native_corenrn_lfp:
                self.ensure_electrode()
            h.cvode_active(0)
            h.steps_per_ms = 1.0 / self.params.sim_dt
            h.dt = self.params.sim_dt
            h.setdt()
            h.cvode.cache_efficient(1)
            h.run()
            self._actual_dt = float(h.dt)

        else:
            self.pc.setup_transfer()
            if getattr(self.params, "enable_lfp", True):
                h.cvode.use_fast_imem(1)
            if getattr(self.params, "enable_lfp", True) and not native_corenrn_lfp:
                self.ensure_electrode()
            parallel_timeout = getattr(self.params, "parallel_timeout", None)
            if parallel_timeout is not None:
                self.pc.timeout(parallel_timeout)
            # h.cvode.cache_efficient(0) # This line causes gap junction Seg Faults
            h.cvode_active(0)
            h.dt = self.params.sim_dt
            self.pc.set_maxstep(1)
            if not getattr(self.params, "legacy_parallel_dt", True):
                h.steps_per_ms = 1.0 / self.params.sim_dt
                h.setdt()
            h.stdinit()
            self._actual_dt = float(h.dt)
            if native_corenrn_lfp:
                self.prepare_corenrn_native_lfp()
            self.pc.psolve(h.tstop)

        if self.mpirank == 0:
            self.write_progress_status(float(h.tstop), float(h.tstop))

        # Clear status updater line
        if self.mpirank == 0:
            if self._status_is_tty:
                print('')

    def write_progress_status(self, current_ms, total_ms):
        """Persist one coarse simulation-progress snapshot for notebook polling."""
        if self.mpirank != 0:
            return

        result_dir = getattr(self, "results_dir", None)
        if not result_dir:
            return

        if total_ms > 0:
            progress = max(0.0, min(float(current_ms) / float(total_ms), 1.0))
            percent = int(progress * 100.0)
        else:
            percent = None

        payload = {
            "current_ms": float(current_ms),
            "total_ms": float(total_ms),
            "percent": percent,
        }
        progress_path = os.path.join(str(result_dir), "sim_progress.json")
        tmp_path = progress_path + ".tmp"
        try:
            with open(tmp_path, "w") as handle:
                json.dump(payload, handle, sort_keys=True)
            os.replace(tmp_path, progress_path)
        except Exception:
            pass

    def print_status(self):
        """
        Emits simulation progress to stdout on TTYs and to a progress file otherwise.
        """

        current_ms = float(self.h.t)
        total_ms = float(getattr(self.params, "tstop", getattr(self.h, "tstop", 0.0)))
        if total_ms > 0:
            progress = max(0.0, min(current_ms / total_ms, 1.0))
            percent = int(progress * 100.0)
            filled = int(progress * 24)
            bar = "#" * filled + "-" * (24 - filled)
            line = "Sim [%s] %3d%% (%.1f / %.1f ms)" % (bar, percent, current_ms, total_ms)
        else:
            percent = None
            line = "Time: %.1f ms" % current_ms

        use_stdout = self._status_mode == "stdout" or (
            self._status_mode not in {"file", "off"} and self._status_is_tty
        )
        if self._status_mode == "off":
            return

        if use_stdout and percent is not None and percent == self._last_status_percent:
            return
        if (not use_stdout) and self._last_status_ms is not None and current_ms <= self._last_status_ms:
            return

        if use_stdout:
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
        else:
            self.write_progress_status(current_ms, total_ms)
        self._last_status_percent = percent
        self._last_status_ms = current_ms

    def setup_status_reporter(self):
        """
        Sets up the NEURON simulation to report the simulation time
        """

        if self.mpirank == 0 and getattr(self.params, "enable_status_report", True):
            h = self.h

            collector_stim = h.NetStim(0.5)
            collector_stim.start = 0
            collector_stim.interval = float(
                os.environ.get(
                    "OBGPU_STATUS_INTERVAL_MS",
                    getattr(self.params, "status_report_interval", 25),
                )
            )
            collector_stim.number = 1e9
            collector_stim.noise = 0

            collector_con = h.NetCon(collector_stim, None)
            collector_con.record(self.print_status)

            self.collector_stim = collector_stim
            self.collector_con = collector_con

    def ensure_electrode(self):
        if self.electrode is None:
            self.electrode = self.create_lfp_electrode(**self._electrode_kwargs)

    def create_lfp_electrode(self, x, y, z, sampling_period, method='Line'):
        """
        Uses the LFPsimpy package to add an LFP electrode at the specified x,y,z location

        See `LFPsimpy package <https://github.com/justasb/LFPsimpy>`_.

        :param x: y, z coordinates in um
        :param sampling_period: How often to compute the LFP signal in ms
        :param method: One of 'Line', 'Point', or 'RC'.
        :return: an LFPsimpy LfpElectrode object
        """

        return ParallelSafeLfpElectrode(x, y, z, sampling_period, method)

    def get_lfp(self):
        """
        Returns the LFP signal in nV

        :return: a tuple of LFP times, and voltages (nV)
        """

        if self.use_corenrn_native_lfp():
            if self.mpirank != 0:
                return [], []

            path = self._native_lfp_report_path or os.path.join(self.get_results_dir(), "lfp_native.tsv")
            lfp_pickle_path = os.path.join(self.get_results_dir(), 'lfp.pkl')
            if not os.path.exists(path):
                if os.path.exists(lfp_pickle_path):
                    with open(lfp_pickle_path, 'rb') as f:
                        return cPickle.load(f)
                raise Exception("CoreNEURON LFP report file was not generated")

            data = np.loadtxt(path, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.size == 0 or data.shape[1] < 2:
                raise Exception("CoreNEURON LFP report file is empty")

            t = data[:, 0].tolist()
            if data.shape[1] == 2:
                lfp = data[:, 1].tolist()
            else:
                lfp = data[:, 1:].sum(axis=1).tolist()

            with open(lfp_pickle_path, 'wb') as f:
                cPickle.dump((t, lfp), f)

            if not getattr(self.params, "keep_native_lfp_debug_files", False):
                for artifact_path in (
                    path,
                    self._native_lfp_report_conf_path,
                    self._native_lfp_sim_conf_path,
                ):
                    if artifact_path and os.path.exists(artifact_path):
                        os.remove(artifact_path)

            return t, lfp

        if self.electrode is None or not any(self.electrode.times):
            raise Exception('Run simulation first to get the LFP')

        t = list(self.electrode.times)
        lfp = list(self.electrode.values)

        if self.nranks > 1:
            all_lfps = self.pc.py_gather((t, lfp), 0)
            if all_lfps is None:
                return t, lfp

            ref_t = np.asarray(all_lfps[0][0], dtype=float)
            summed_lfp = np.zeros_like(ref_t)
            for rank_t, rank_lfp in all_lfps:
                rank_t = np.asarray(rank_t, dtype=float)
                rank_lfp = np.asarray(rank_lfp, dtype=float)
                if rank_t.shape != ref_t.shape or not np.allclose(rank_t, ref_t, atol=1e-9, rtol=0):
                    raise RuntimeError("LFP sample times diverged across MPI ranks")
                summed_lfp += rank_lfp

            t = ref_t.tolist()
            lfp = summed_lfp.tolist()

        if self.mpirank == 0:
            self.ensure_results_dir()
            with open(os.path.join(self.results_dir, 'lfp.pkl'), 'wb') as f:
                cPickle.dump((t, lfp), f)

        return t, lfp

    def get_model_inputsegs(self):
        """
        Queries the model database to get the 'root' segments of the tufted dendrites
        of the mitral and tufted cells

        :return: A dict that maps the cell model's class name to the name of the root tufted dendrite section
        """

        if self._model_inputsegs_cache is not None:
            return self._model_inputsegs_cache

        # Get all the different cell models used in the slice
        input_models = set()
        for cells in self.glom_cells.values():
            for cell in cells:
                input_models.add(cell[:cell.find('[')])

        # Get each model's input segments (in the tuft)
        model_inputsegs = {m.class_name: m.tufted_dend_root
                           for m in CellModel \
                               .select(CellModel.class_name, CellModel.tufted_dend_root) \
                               .where(CellModel.class_name.in_(list(input_models)))}

        self._model_inputsegs_cache = model_inputsegs
        return model_inputsegs

    def get_odor_glom_intensities(self, odor):
        cached = self._odor_glom_intensities_cache.get(odor)
        if cached is None:
            cached = {g.glom_id: g.intensity
                      for g in OdorGlom
                          .select(OdorGlom.glom_id, OdorGlom.intensity)
                          .join(Odor)
                          .where(Odor.name == odor)}
            self._odor_glom_intensities_cache[odor] = cached
        return cached

    def get_glom_input_seg_cache(self):
        if self._glom_input_seg_cache is not None:
            return self._glom_input_seg_cache

        model_inputsegs = self.get_model_inputsegs()
        cache = {}

        for glom_id, cells in self.glom_cells.items():
            input_segs = []
            for cell in cells:
                rank_cell = self.rank_section_name(cell)

                # Add inputs only to cells that are on this rank
                if rank_cell is None:
                    continue

                model_class = rank_cell[:rank_cell.find('[')]
                input_seg = model_inputsegs[model_class]
                seg_address = 'h.' + rank_cell + '.' + input_seg

                single_rank_address = 'h.' + cell + '.' + input_seg
                single_rank_gid = self.stable_hash(single_rank_address)

                input_segs.append((seg_address, single_rank_gid, single_rank_address))

            cache[int(glom_id)] = input_segs

        self._glom_input_seg_cache = cache
        return cache

    def get_gap_junction_seg_cache(self, in_name):
        cached = self._gap_junction_seg_cache.get(in_name)
        if cached is not None:
            return cached

        model_inputsegs = self.get_model_inputsegs()
        cache = {}

        for glom_id, cells in self.glom_cells.items():
            input_segs = []
            for cell in cells:
                if in_name not in cell:
                    continue

                model_class = cell[:cell.find('[')]
                input_seg = model_inputsegs[model_class]

                single_rank_address = 'h.' + cell + '.' + input_seg
                single_rank_gid = self.stable_hash(single_rank_address)

                rank_cell = self.rank_section_name(cell)
                if rank_cell is not None:
                    seg_address = 'h.' + rank_cell + '.' + input_seg
                else:
                    seg_address = None

                input_segs.append((seg_address, single_rank_gid))

            cache[glom_id] = input_segs

        self._gap_junction_seg_cache[in_name] = cache
        return cache

    def add_gap_junctions(self, in_name, g_gap):
        """
        Adds gap junctions between tufted dendrites of specified cells

        :param in_name: A part of a cell class name (e.g. 'Mitral') used to select a cell to which the GJ is added
        :param g_gap: The conductance of the gap junctions
        """

        if g_gap <= 0:
            return

        for glom_id, input_segs in self.get_gap_junction_seg_cache(in_name).items():
            if len(input_segs) > 0:
                self.create_gap_junctions_between(input_segs, g_gap)

    def create_gap_junctions_between(self, input_segs, g_gap):
        """
        Creates gap junctions between a list of specified segments. GJs are connected in a chain
        (e.g. Seg1 <-GJ1-> Seg2 <-GC2-> Seg3)

        :param input_segs: List of segments to connect by gap junctions
        :param g_gap: Gap junction conductance
        """

        count = len(input_segs)

        if count < 2:
            return

        h = self.h

        first_seg = input_segs[0]
        last_seg = input_segs[-1]

        if count > 2:
            for i, seg in enumerate(input_segs[:-1]):
                next_seg = input_segs[i + 1]

                self.create_gap_junction(seg, next_seg, g_gap)

        self.create_gap_junction(first_seg, last_seg, g_gap)

    def create_gap_junction(self, seg_1_info, seg_2_info, g_gap):
        """
        Creates a gap junction between two segments

        :param seg_1_info: Tuple of the name and gid of the first segment
        :param seg_2_info: Tuple of the name and gid of the second segment
        :param g_gap: Gap junction conductance
        """

        h = self.h

        seg_1_name, seg_1_gid = seg_1_info
        seg_2_name, seg_2_gid = seg_2_info

        if seg_1_name is not None:
            seg1 = self.resolve_segment(seg_1_name)
            self.remember_cell_source_gid(seg_1_name.replace("h.", ""), seg_1_gid)

            if seg_1_gid not in self.gj_source_gids:
                self.pc.source_var(seg1._ref_v, seg_1_gid, sec=seg1.sec)
                self.gj_source_gids.add(seg_1_gid)

            gap1 = h.GapJunction(seg1.x, sec=seg1.sec)
            gap1.g = g_gap
            self.pc.target_var(gap1, gap1._ref_v_other, seg_2_gid)
            self.gjs.append(gap1)

        if seg_2_name is not None:
            seg2 = self.resolve_segment(seg_2_name)
            self.remember_cell_source_gid(seg_2_name.replace("h.", ""), seg_2_gid)

            if seg_2_gid not in self.gj_source_gids:
                self.pc.source_var(seg2._ref_v, seg_2_gid, sec=seg2.sec)
                self.gj_source_gids.add(seg_2_gid)

            gap2 = h.GapJunction(seg2.x, sec=seg2.sec)
            gap2.g = g_gap
            self.pc.target_var(gap2, gap2._ref_v_other, seg_1_gid)
            self.gjs.append(gap2)


    def add_inputs(self, odor, t, rel_conc):
        """
        Add odor stimulation to the tufts of the principal cells

        :param odor: The name of the odor
        :param t: Onset time
        :param rel_conc: Relative concentration 0-1
        """

        # Get input odor glomeruli
        glom_intensities = self.get_odor_glom_intensities(odor)

        for glom_id, input_segs in self.get_glom_input_seg_cache().items():
            if len(input_segs) > 0:
                glom_intensity = glom_intensities[glom_id] * rel_conc
                self.stim_glom_segments(t, input_segs, glom_intensity)

    def add_stimulus_inputs(self, t, input_spec, intensity=1.0, cell_types=None):
        """
        Stimulate glomerular tuft segments using a custom InputSpec.

        Targets the same tuft segments as odor inputs (all glomeruli, both MC
        and TC by default).  Glomerulus-specific intensity weighting is not
        applied; every glomerulus receives the same flat ``intensity``.

        :param t: Onset time in ms
        :param input_spec: InputSpec instance defining spike generation
        :param intensity: Scaling factor 0-1 passed to the spec
        :param cell_types: Optional list e.g. ['MC'] to restrict targeting;
                           None stimulates both MC and TC tufts
        """
        for glom_id, input_segs in self.get_glom_input_seg_cache().items():
            if not input_segs:
                continue
            if cell_types is not None:
                input_segs = [
                    seg for seg in input_segs
                    if any(ct in seg[0] for ct in cell_types)
                ]
            if input_segs:
                self.stim_glom_segments(t, input_segs, intensity, input_spec=input_spec)

    def load_glom_cells(self):
        """
        Loads a dict that maps glomeruli ids to cells that are attached to each glomerulus
        """

        with open(os.path.join(self.slice_dir, 'glom_cells.json')) as f:
            self.glom_cells = json.load(f)

    def get_gaussian_spike_train(self, spikes=50, start_time=100, duration=10, rng=None):
        """
        Gets a spike train from a gaussian probability distribution whose 99% range starts
        at the specified time and lasts for the specified duration.

        :param spikes: The number of spikes to generate
        :param start_time: The onset time of the gaussian
        :param duration: The duration of the gaussian
        :return: A numpy array of spike times in chronological order
        """

        # Create a gaussian whose 99% range starts at start_time
        # and ends at start_time + duration
        normal_stdev = duration / (2.576 * 2)

        if rng is None:
            rng = np.random

        times = rng.normal(start_time + (duration / 2.0), normal_stdev, spikes)

        # Remove any spikes outside this range
        times = times[np.where((times > start_time) & (times < start_time + duration))]
        times.sort()

        return times

    def load_cells(self, cell_type):
        """
        Load the cells of the specified type onto least busy MPI ranks.

        'Busyness' of a rank is the sum of all cell complexities on that rank, as measured by the number
        of segments of each cell.

        :param cell_type: One of 'MC', 'GC', 'TC'
        """

        # Load the cell json file
        path = os.path.join(self.slice_dir, cell_type + 's.json')

        with open(path, 'r') as f:
            group_dict = json.load(f)

        # Count how many of each cell model will be on each rank
        rank_cell_counts = {r: {} for r in range(self.nranks)}

        for ri, root in enumerate(group_dict['roots']):
            # Get the least loaded rank
            min_complexity, min_complexity_rank = heappop(self.rank_complexities)

            # Cell nseg count is used as a proxy for complexity
            model_name = root['name']
            model_name = model_name[0:model_name.find('[')]
            nsegs = self._model_nseg_count_cache.get(model_name)
            if nsegs is None:
                nsegs = self.get_nseg_count(root)
                self._model_nseg_count_cache[model_name] = nsegs

            # Add to rank complexity and push back onto the heap
            heappush(self.rank_complexities, (min_complexity + nsegs, min_complexity_rank))

            # Assign cell to least busy rank
            cell_rank = min_complexity_rank

            name = model_name

            count = rank_cell_counts[cell_rank].get(name, 0)

            self.mpimap[root['name'][:root['name'].find(']') + 1]] = {
                'name': name + '[' + str(count * 2) + ']',
                'rank': cell_rank
            }

            count += 1
            rank_cell_counts[cell_rank][name] = count

        # Load that many base instances of each model
        self.cells[cell_type] = []
        for cell_model_name, count in rank_cell_counts[self.mpirank].items():
            cell_factory = CELL_MODEL_FACTORIES[cell_model_name]
            cell_models = [cell_factory() for _ in range(count)]
            self.cells[cell_type].extend(cell_models)

        return group_dict

    def finish_loading_cells(self, group_dicts):
        # Update section index once after all base cells exist on this rank.
        self.bn_server.update_section_index()

        # Initialize BlenderNEURON groups once, then apply the saved cell json.
        self.bn_server.init_mpi(self.pc, self.mpimap)
        self.bn_server.update_groups(group_dicts)

    def record_from_somas(self, cell_type):
        """
        Adds NEURON vector recorders to the somas of the specified cell types

        :param cell_type: One of 'MC', 'GC', 'TC'
        """

        h = self.h

        for cell_model in self.cells[cell_type]:
            v_vec = h.Vector()
            if self._use_dense_soma_recording:
                v_vec.record(cell_model.soma(0.5)._ref_v, sec=cell_model.soma)
            else:
                v_vec.record(cell_model.soma(0.5)._ref_v, self.params.recording_period)
            self.v_vectors[str(cell_model.soma)] = v_vec

    def save_recorded_vectors(self):
        """
        Saves soma voltage traces and odor input spike times to Pickle files for later processing

        Saves soma traces as a compressed artifact plus input_times.pkl.
        """

        # Gather cell voltage vectors
        all_v_vecs = self.pc.py_gather(self.v_vectors, 0)

        if all_v_vecs is not None:
            self.ensure_results_dir()
            result = []
            sampled_t = self.t_vec.to_python()
            dense_t = None
            for rank_v_vecs in all_v_vecs:
                for cell, v_vec in rank_v_vecs.items():
                    values = v_vec.to_python()
                    if self._use_dense_soma_recording:
                        downsampled_v = self._downsample_trace(values)
                        if dense_t is None:
                            dense_t = np.arange(
                                0.0,
                                float(self.h.tstop),
                                float(self.params.recording_period),
                            ).tolist()
                        if len(downsampled_v) != len(dense_t):
                            target_len = min(len(downsampled_v), len(dense_t))
                            cell_t = dense_t[:target_len]
                            downsampled_v = downsampled_v[:target_len]
                        else:
                            cell_t = dense_t
                        result.append((cell, cell_t, downsampled_v.tolist()))
                    else:
                        result.append((cell, sampled_t, values))

            trace_format = getattr(self.params, "soma_trace_format", DEFAULT_SOMA_TRACE_FORMAT)
            trace_dtype = getattr(self.params, "soma_trace_dtype", DEFAULT_SOMA_TRACE_DTYPE)
            save_soma_spike_artifact(
                result,
                self.results_dir,
                threshold=getattr(self.params, "soma_spike_threshold", DEFAULT_SOMA_SPIKE_THRESHOLD_MV),
                min_prominence_mv=getattr(
                    self.params,
                    "soma_spike_min_prominence_mv",
                    DEFAULT_SOMA_SPIKE_MIN_PROMINENCE_MV,
                ),
                refractory_ms=getattr(
                    self.params,
                    "soma_spike_refractory_ms",
                    DEFAULT_SOMA_SPIKE_REFRACTORY_MS,
                ),
            )
            save_voltage_summary_artifact(result, self.results_dir)
            save_path = save_soma_trace_artifact(
                result,
                self.results_dir,
                trace_format=trace_format,
                trace_dtype=trace_dtype,
            )
            legacy_path = os.path.join(self.results_dir, SOMA_TRACE_FILENAME_PKL)
            compressed_path = os.path.join(self.results_dir, SOMA_TRACE_FILENAME_NPZ)
            stale_path = legacy_path if os.path.basename(str(save_path)) == SOMA_TRACE_FILENAME_NPZ else compressed_path
            if os.path.exists(stale_path):
                os.unlink(stale_path)

        # Gather input event time vectors
        all_input_vecs = self.pc.py_gather(self.input_vectors, 0)

        if all_input_vecs is not None:
            self.ensure_results_dir()
            result = []
            for rank_input_vecs in all_input_vecs:
                for seg_name, t_vec in rank_input_vecs:
                    result.append((seg_name, t_vec.to_python()))

            with open(os.path.join(self.results_dir, 'input_times.pkl'), 'wb') as f:
                cPickle.dump(result, f)

        all_gc_output_vecs = self.pc.py_gather(self.gc_output_event_vectors, 0)

        if all_gc_output_vecs is not None:
            self.ensure_results_dir()
            result = []
            for rank_event_vecs in all_gc_output_vecs:
                for event_meta, t_vec in rank_event_vecs:
                    record = dict(event_meta)
                    record["times"] = t_vec.to_python()
                    result.append(record)

            with open(os.path.join(self.results_dir, 'gc_output_events.pkl'), 'wb') as f:
                cPickle.dump(result, f)

    def _downsample_trace(self, values):
        values = np.asarray(values, dtype=float)
        if values.size < 2:
            return values

        period = float(self.params.recording_period)
        actual_dt = float(getattr(self, "_actual_dt", self.params.sim_dt))
        stride = period / actual_dt
        nearest_stride = round(stride)
        if nearest_stride <= 1 or abs(stride - nearest_stride) > 1e-9:
            return values

        return values[::int(nearest_stride)]

    def get_nseg_count(self, root_dict):
        """
        Recursively counts the number of segments of a cell provided its BlenderNEURON root segment dict

        :param root_dict: The root segment dict of a cell as saved by BlenderNEURON
        :return: The total number of segments of the cell
        """

        count = root_dict["nseg"]

        for child_dict in root_dict['children']:
            count += self.get_nseg_count(child_dict)

        return count

    def load_synapse_set(self, synapse_set):
        """
        Uses BlenderNEURON to load a previously saved set of synapses between a population of cells

        :param synapse_set: One of 'GCs__MCs' or 'GCs__TCs' as seen in the olfactorybulb.slices.DorsalColumnSlice folder.
        """

        path = os.path.join(self.slice_dir, synapse_set + '.json')

        with open(path, 'r') as f:
            synapse_set_dict = json.load(f)

        self.bn_server.create_synapses(synapse_set_dict)

    def add_gc_kar_synapse_set(self, synapse_set):
        """
        Add optional MC/TC->GC kainate receptors at existing reciprocal excitation sites.

        The saved reciprocal synapse JSON names the GC side as ``source`` and
        the MC/TC side as ``dest`` because GC->MC/TC GABA is the forward
        connection in that file. For the excitatory reciprocal direction, the
        presynaptic glutamate source is therefore the saved ``dest_section``
        and the postsynaptic GC target is the saved ``source_section``.
        """

        gmax = float(getattr(self.params, "kar_gc_gmax", 0.0))
        if gmax <= 0:
            return

        self.require_mechanism("KainateSyn")
        h = self.h

        path = os.path.join(self.slice_dir, synapse_set + '.json')
        with open(path, 'r') as f:
            entries = json.load(f)["entries"]

        for entry in entries:
            if not entry.get("is_reciprocal", False):
                continue

            target_section_name = self.rank_section_name(entry["source_section"])
            source_section_name = self.rank_section_name(entry["dest_section"])
            syn_on_rank = target_section_name is not None
            source_on_rank = source_section_name is not None

            if not syn_on_rank and not source_on_rank:
                continue

            source_gid = self.bn_server.segment_gid(
                entry["dest_section"],
                entry["dest_seg_i"],
                False,
            )

            if source_on_rank:
                source_seg = self.resolve_segment(
                    "h.%s(%s)" % (source_section_name, float(entry.get("dest_x", 0.5)))
                )
                if self.pc.gid_exists(source_gid) == 0:
                    self.bn_server.assign_gid_to_source_seg(
                        source_seg.sec,
                        float(entry.get("dest_x", 0.5)),
                        float(entry.get("threshold", 0.0)),
                        source_gid,
                    )
                self.bn_server.remember_cell_source_gid(source_section_name, source_gid)

            if not syn_on_rank:
                continue

            target_seg = self.resolve_segment(
                "h.%s(%s)" % (target_section_name, float(entry.get("source_x", 0.5)))
            )
            syn = h.KainateSyn(target_seg)
            self.configure_kar_synapse(syn, gmax)

            netcon = self.pc.gid_connect(source_gid, syn)
            netcon.delay = float(entry.get("delay", 0.5))
            netcon.weight[0] = float(entry.get("weight", 1.0)) * float(
                getattr(self.params, "kar_gc_weight_scale", 1.0)
            )
            self.gc_kar_synapses.append((syn, netcon))

    def record_gc_output_events(self):
        h = self.h

        for synapse_set_name in ['GCs__MCs', 'GCs__TCs']:
            synapses = self.bn_server.synapse_sets.get(synapse_set_name, [])
            if not synapses:
                continue

            path = os.path.join(self.slice_dir, synapse_set_name + '.json')
            with open(path, 'r') as f:
                entries = json.load(f)["entries"]

            for entry, synapse_parts in zip(entries, synapses):
                netcon = synapse_parts[0]
                syn = synapse_parts[1]
                if netcon is None or syn is None:
                    continue
                if entry.get("dest_syn") != "GabaSyn":
                    continue

                event_vec = h.Vector()
                netcon.record(event_vec)
                self.gc_output_event_vectors.append((
                    {
                        "set_name": synapse_set_name,
                        "source_section": entry["source_section"],
                        "dest_section": entry["dest_section"],
                        "source_x": float(entry.get("source_x", 0.5)),
                        "dest_x": float(entry.get("dest_x", 0.5)),
                        "weight": float(entry.get("weight", 0.0)),
                        "delay": float(entry.get("delay", 0.0)),
                        "threshold": float(entry.get("threshold", 0.0)),
                    },
                    event_vec,
                ))
