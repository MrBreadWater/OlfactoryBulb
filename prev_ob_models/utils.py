import os, inspect
from pathlib import Path
import platform
debug = False

class RunInClassDirectory:
    '''
    A utility class that temporarily switches the os working dir to the directory of the file in which the passed in
    class is located. The working dir is restored after the class is closed. The class should be used using the "with"
    statement e.g.:

    with RunInClassDirectory(NeuronCellClass):
        from neuron import h # <--- this will be executed in NeuronCellClass's folder
        etc...

    '''
    def __init__(self, the_class):
        self.the_class = the_class

    def __enter__(self):
        self.cur_dir = os.getcwd()
        class_file = inspect.getfile(self.the_class)
        class_dir = os.path.join(os.getcwd(), os.path.dirname(class_file))

        if debug:
            print('Temporarily changing directory to', class_dir)

        os.chdir(class_dir)

    def __exit__(self, exc_type, exc_value, traceback):
        if debug:
            print('Restoring working dir to', self.cur_dir)

        os.chdir(self.cur_dir)


class IsolatedCell(object):
    def close_window(self, name_contains):
        """
        Closes a NEURON window that matches the parameter string

        :param name_contains: The string to match in window name
        :return: Nothing
        """
        if not hasattr(self,"pwm"):
            from neuron import h
            self.pwm = h.PWManager()

        target_window_index = next(i
                                   for i in range(int(self.pwm.count()))
                                   if name_contains in self.pwm.name(i))

        self.pwm.close(target_window_index)


_MECH_LOAD_CACHE = {}
_LOADED_MECH_BASES = set()


def _call_load_mechanisms(load_mechanisms, path):
    try:
        return load_mechanisms(path, warn_if_already_loaded=False)
    except TypeError:
        return load_mechanisms(path)


def load_mechanisms_from_candidates(load_mechanisms,
                                   anchor_path,
                                   mechanism_dir_name="Mechanisms",
                                   sentinel_mechanisms=None):
    """
    Load compiled NEURON mechanisms by searching common build layouts.

    Legacy setups often compile inside ``Mechanisms/`` while newer project-level
    builds place ``aarch64/`` or ``x86_64/`` at the repo root. Some builds keep
    the actual shared library under ``<arch>/.libs/`` while others place it
    directly under ``<arch>/``. This helper accepts either layout and returns
    the directory passed to ``neuron.load_mechanisms``.
    """

    anchor = Path(anchor_path).resolve()
    key = (str(anchor), mechanism_dir_name)
    cached = _MECH_LOAD_CACHE.get(key)
    if cached is not None:
        return cached

    if sentinel_mechanisms:
        from neuron import h

        if all(hasattr(h, mech_name) for mech_name in sentinel_mechanisms):
            loaded = "already-loaded"
            _MECH_LOAD_CACHE[key] = loaded
            return loaded

    libname = "libnrnmech.so"
    arch_names = []
    machine = platform.machine()
    if machine:
        arch_names.append(machine)
    arch_names.extend(["aarch64", "x86_64", "i686", "powerpc", "umac"])

    candidates = []
    for parent in [anchor, *anchor.parents]:
        candidates.append(parent / mechanism_dir_name)
        candidates.append(parent)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidate_str = str(candidate)
        if candidate_str in _LOADED_MECH_BASES:
            _MECH_LOAD_CACHE[key] = candidate_str
            return candidate_str
        for arch in arch_names:
            arch_dir = candidate / arch
            lib_candidates = (
                arch_dir / libname,
                arch_dir / ".libs" / libname,
            )
            if any(path.exists() for path in lib_candidates):
                try:
                    if _call_load_mechanisms(load_mechanisms, candidate_str):
                        _LOADED_MECH_BASES.add(candidate_str)
                        _MECH_LOAD_CACHE[key] = candidate_str
                        return candidate_str
                except RuntimeError as exc:
                    if "already exists" in str(exc):
                        _LOADED_MECH_BASES.add(candidate_str)
                        _MECH_LOAD_CACHE[key] = candidate_str
                        return candidate_str
                    raise

    # Final fallback preserves the old behavior for environments that can
    # discover the mechanisms through the working directory.
    if _call_load_mechanisms(load_mechanisms, mechanism_dir_name):
        _LOADED_MECH_BASES.add(mechanism_dir_name)
        _MECH_LOAD_CACHE[key] = mechanism_dir_name
        return mechanism_dir_name

    raise FileNotFoundError(
        f"Could not find compiled NEURON mechanisms near {anchor} using '{mechanism_dir_name}'. "
        "Run nrnivmodl and ensure the resulting architecture directory is present."
    )
