# Project notes for Claude

Read `docs/HANDOFF.md` first. It holds the project's state across sessions:
what each run did, where the monitor, dashboard, and pods run, how to check on
the active run, and what's left to do.

Rules for this project:

- Don't stop, restart, arm, or change a training run, and don't create or
  delete a Runpod pod, unless the owner asks in this session.
- Never print values from `.env`.
- In shell commands, call `/bin/ls`, never `ls` or `eza`.
- Keep `docs/HANDOFF.md` current when run state or infrastructure changes.
