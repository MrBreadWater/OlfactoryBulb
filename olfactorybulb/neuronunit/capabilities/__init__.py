# MOCKS for autodoc
import quantities as pq
if getattr(getattr(pq, "mV", None).__class__, "__module__", "") == "sphinx.ext.autodoc.importer":
    pq.mV = pq.ms = pq.Hz = 1
# END MOCKS

import quantities as pq
import sciunit


class SupportsVoltageClamp(sciunit.Capability):
    """Indicates that the model can be held at three levels of voltages using a voltage clamp"""

    def clamp_voltage(self, voltages=[0*pq.mV]*3, durations=[0*pq.ms]*3):
        '''
        Maintains the model membrane potential for the specified durations at the specified voltages

        :param voltages: a 3-element array of voltages to clamp to
        :param durations: a 3-element array of durations to maintain the corresponding voltage levels
        :return: neo.core.AnalogSignal of the current required to keep the model at the specified voltages
        '''
        raise NotImplementedError()


class SupportsSettingTemperature(sciunit.Capability):
    """Indicates that the model can be executed using a specific temperature in Celsius"""

    def set_temperature(self, temperature=6.3):
        '''
        Specifies the simulator temperature

        :param temperature: the simulator temperature in degrees Celsius
        :return: Nothing
        '''
        raise NotImplementedError()


class SupportsSettingStopTime(sciunit.Capability):
    """Indicates that the model's simulation stop time can be specified"""

    def set_stop_time(self, tstop):
        '''
        Specifies the simulator stop time

        :param temperature: the simulator stop time in ms
        :return: Nothing
        '''
        raise NotImplementedError()


class ProvidesMetricSummary(sciunit.Capability):
    """Indicates that a model can expose summarized validation metrics."""

    def get_metric_summary(self, group, metric_key, unit_text=""):
        """
        Return the summarized metric value for one validation group and key.

        Implementations may return either a naked float or a quantities value.
        """
        raise NotImplementedError()


class ProvidesMetricRows(sciunit.Capability):
    """Indicates that a model can expose per-row validation metrics."""

    def get_metric_value_map(self, metric_key, entity_key="cell_name", unit_text=""):
        """
        Return a mapping from audited-row identity to the requested metric value.

        The maintained runtime/model seam may return a typed frozen mapping
        payload rather than a fresh mutable dict. Values may be naked floats or
        quantities values.
        """
        raise NotImplementedError()


class ProvidesProtocolEvidenceRows(sciunit.Capability):
    """Indicates that a model can expose protocol-evidence row collections."""

    def get_protocol_evidence_rows(self, evidence_key):
        """
        Return the row collection associated with one protocol-evidence key.
        """
        raise NotImplementedError()


class ProvidesProtocolEvidenceMap(sciunit.Capability):
    """Indicates that a model can expose the full protocol-evidence payload."""

    def get_protocol_evidence_map(self):
        """
        Return the protocol-evidence mapping for the current validation run.
        """
        raise NotImplementedError()


class ProvidesProtocolEvidenceBundle(sciunit.Capability):
    """Indicates that a model can expose the typed protocol-evidence bundle."""

    def get_protocol_evidence_bundle(self):
        """
        Return the typed protocol-evidence bundle for the current validation run.
        """
        raise NotImplementedError()
