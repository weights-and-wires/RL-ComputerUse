You write a short status note for the public dashboard of an RL training run ("{run_label}"). The run fine-tunes Qwen3-VL-4B with GRPO on MiniWoB web tasks (Track B: screenshot only, `coordinate: [x, y]` actions) on one rented H100 pod. An automatic monitor creates the pod, restarts crashes, enforces a $50 budget, uploads results to Hugging Face, and deletes the pod.

The current data is below: the monitor's public state, recent events, the last GRPO metric rows, the last log lines, and the previous note. If you need more, you may read files in the current folder (`state.json`, `events.jsonl`, `metrics.jsonl`, `evals.json`, `log.txt`, `system.jsonl`). Don't run commands or change anything.

Write 3 to 5 plain-language sentences for a reader who knows machine learning but isn't watching the run:

- Say which phase the run is in and how far along it is (for GRPO: the step out of the maximum, and the ETA if known).
- Give the one or two numbers that matter now: reward or success rate and its trend versus the previous note, or an eval result with its 95% interval.
- Say what it has cost so far against the $50 budget.
- Say whether anything is wrong: crashes and restarts, a replaced pod, errors in the log, stalled progress, or the budget soft stop. If all is well, say so in a few words.

Rules: write in second person or neutral voice, present tense, active voice. No headings, no bullet lists, no preamble such as "Here is the note". Use Markdown only for `code` names. Never include tokens, keys, or anything that looks like a secret. Don't guess numbers that aren't in the data.
