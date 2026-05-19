# Repository Agent Notes

- When making user-requested repository changes, create a targeted git commit before the final response unless the user explicitly says not to commit.
- Stage only files that belong to the current task. Leave unrelated dirty files and user changes unstaged.
- For notebooks, avoid committing transient execution output unless the output is intentionally part of the deliverable.
