"""Shared helpers for remote Slurm command wrappers.

These scripts are executed directly on remote hosts, so keep imports limited to
the Python standard library and syntax compatible with older remote Python
interpreters.
"""

import shlex


def shell_join(parts):
    """Portable equivalent of ``shlex.join`` for older remote Python stacks."""
    return " ".join(shlex.quote(str(part)) for part in parts)


def path_is_within(path_value, root_value):
    """Return whether one string path is equal to or nested under another."""
    root_text = str(root_value).rstrip("/")
    path_text = str(path_value)
    if not root_text:
        return False
    return path_text == root_text or path_text.startswith(root_text + "/")


def normalize_sbatch_args(values):
    """Normalize raw sbatch args so split flag/value pairs become one directive."""
    normalized = []
    index = 0
    values = [str(value) for value in values]
    while index < len(values):
        current = values[index]
        if current.startswith("-") and "=" not in current and index + 1 < len(values):
            next_value = values[index + 1]
            if not next_value.startswith("-"):
                normalized.append("{} {}".format(current, next_value))
                index += 2
                continue
        normalized.append(current)
        index += 1
    return normalized


def slurm_directives(args, job_name):
    """Return ``#SBATCH`` header lines for one generated Slurm script."""
    directives = ["#SBATCH --job-name={}".format(job_name[:120])]
    if args.partition:
        directives.append("#SBATCH --partition={}".format(args.partition))
    if args.account:
        directives.append("#SBATCH --account={}".format(args.account))
    if args.time:
        directives.append("#SBATCH --time={}".format(args.time))
    if args.gpus is not None:
        directives.append("#SBATCH --gpus={}".format(args.gpus))
    if args.cpus_per_task is not None:
        directives.append("#SBATCH --cpus-per-task={}".format(args.cpus_per_task))
    if args.mem:
        directives.append("#SBATCH --mem={}".format(args.mem))
    for extra in normalize_sbatch_args(args.sbatch_arg):
        directives.append("#SBATCH {}".format(extra))
    return directives


def requested_mpi_rank_count(command):
    """Return the requested MPI rank count from one command list, if present."""
    options_with_values = {"-n", "-np", "--np", "--ntasks", "--ntasks-per-job"}
    for index, part in enumerate(command):
        if part in options_with_values and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                continue
        for prefix in ("-n", "-np"):
            suffix = part[len(prefix):]
            if part.startswith(prefix) and suffix:
                try:
                    return int(suffix)
                except ValueError:
                    pass
        for prefix in ("--ntasks=", "--ntasks-per-job="):
            if part.startswith(prefix):
                try:
                    return int(part.split("=", 1)[1])
                except ValueError:
                    pass
    return None
