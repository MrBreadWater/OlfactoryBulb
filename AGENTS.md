# Repository Agent Notes

- When making user-requested repository changes, create a targeted git commit before the final response unless the user explicitly says not to commit.
- Stage only files that belong to the current task. Leave unrelated dirty files and user changes unstaged.
- For notebooks, avoid committing transient execution output unless the output is intentionally part of the deliverable.
- When changing a notebook-facing default, verify the actual user paths that should observe it: single run, remote run, sweep batch, live sync, final sync, and load/animation helpers. Add or update tests for the default behavior, not only the helper implementation.
- In this workspace, `/home/michael/OlfactoryBulb` is the user-facing checkout path and may be a symlink to `/home/alek/OlfactoryBulb`. For notebook/Jupyter work, preserve Michael's authenticated session and prefer `/home/michael/OlfactoryBulb` in user-facing paths.
