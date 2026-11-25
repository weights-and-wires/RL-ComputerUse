# Project handoff

This page gives a new session everything it needs to pick up the project: what
each run did, where things run, how to check on them, and what's left to do.
The facts are as of 2026-11-25, 12:00 UTC. Before you act on run state, check
the live status with the commands in [Check on run 4](#check-on-run-4).

## The project in one paragraph

You RL-post-train `Qwen/Qwen3-VL-4B-Instruct` with GRPO and LoRA to do
MiniWoB++ browser tasks (BrowserGym, headless Chromium, vLLM rollouts, the
custom trainer in `cua/nano_grpo/`). Track A clicks elements by ID; Track B
sees only a screenshot and clicks by pixel (`coordinate: [x, y]`). Runs 1 and 2
trained Track A. Runs 3 and 4 train Track B. The owner dropped Track A after
run 3, because modern computer-use agents act on pixels. **Run 4 is the final
run.** The owner doesn't plan to spend more after it, and the model stays at
4B parameters.

## Runs at a glance

| Run | Track | GPU | Cost | Headline |
|---|---|---|---|---|
| 1 | A | H100 | about $26 | Practice set 27.3% to 57.8%; held-out 55.6% to 62.8% |
| 2 | A | A100 | about $15 | Run-2 practice set 37.8% to 72.6%; held-out flat (+0.7) |
| 3 | B | Secure H100 SXM, $4.39/h | $44.80 | Pool 51.1% to 79.4% (base 39.2%); held-out 63.1% to 71.5% (+8.4, 95% range +6.4 to +10.4); Track A dropped 5.5 points |
| 4 | B | Secure A100 SXM, $2.19/h | running | See [Run 4](#run-4) |

The plans with the reasoning behind each run are in `docs/PLAN.md`,
`docs/PLAN_RUN2.md`, `docs/PLAN_RUN3.md`, and `docs/PLAN_RUN4.md`.

### Run 3 findings

- 240 GRPO steps in 8.1 h, 1,420 trained (mixed) groups. The $42 soft stop
  ended GRPO; evals, upload, and pod deletion ran on their own.
- Held-out gains, none negative: read-table-2 +27, email-inbox-delete +21,
  click-checkboxes-large +15, count-sides +10.
- The pool saturated: the zero-variance group share went from 0.47 to 0.70, and
  dynamic sampling needed 2.8 of its 3 rounds per step by the end.
- Rollouts became the bottleneck: about 80 s per step, against about 28 s of
  training.
- At step 207, one update had an importance ratio of 3.2e6, KL 169, and a
  gradient norm of 3,710. It recovered by the next step.
- Track A forgetting, scored on run 2's pool: email-inbox-star-reply −30,
  choose-list −22, social-media −14.
- The pool was picked from run 2's screen, which flatters the pool comparison a
  little. The held-out comparison has no such bias.
- Results: `results/run3/` locally (git-ignored), and on the Hub under
  `results/run3/`. The adapter is `checkpoints/grpo-4b-run3/final` in the Hub
  repo `HF_ADAPTER_REPO`.

## Run 4

Run 4 started from the run-3 adapter on 2026-11-25 at 06:23 UTC on pod
`nn6wp7c72pfion`. It's screen-only (Track B). The plan is `docs/PLAN_RUN4.md`.

What's new in run 4:

- Episodes run up to 20 turns, with per-task budgets. The prompt restarts every
  4 turns with a text log of earlier actions, so context stays small.
- Track B adds `double_click` and `left_click_drag`.
- The loss caps the trainer/vLLM probability ratio, uses a dual clip, and drops
  episodes whose log-probabilities disagree with vLLM's.
- Metric rows carry `"kind": "train"` or `"kind": "eval"`.
- A full-suite eval scores 117 tasks at 20 seeds before and after GRPO.
- `post_eval` keeps the best mid-training adapter if it beats the final one by
  more than 2 points.
- stock-market is excluded: its price moves faster than the agent can act.
- sign-agreement stays in held-out. 42 of its 100 seeds need no scrolling, and
  those are exactly the ones every earlier model solved.

What happened so far:

- No Community H100 was in stock, so the owner chose Secure A100 SXM at
  $2.19/h with a 12 h GRPO cap. `~/cua-run4/config.json` on Bazzite sets both
  (`gpu_order` and `pod_env`).
- The smoke test passed in 11 minutes. 64 browsers crashed 113 times, so the
  pipeline fell back to 32 browsers, as designed. The cause isn't verified;
  it's likely container CPU or shared-memory limits.
- Drag and double-click reached the browser without errors in the smoke test.
  Whether the model can learn drags is the least certain part of the run.
- Phases: smoke 06:28-06:39, screen and select to 07:01, pre_eval to 08:31,
  then GRPO from 08:31.
- At 12:00 UTC: GRPO step 85, checkpoint 80 synced, spend $17.20 of $50, no
  restarts.

Run 4's pool: 18 learn tasks (use-slider-2, drag-items-grid, drag-box,
drag-shapes-2, email-inbox, social-media, email-inbox-star-reply, guess-number,
tic-tac-toe, click-scroll-list, form-sequence, email-inbox-noscroll,
enter-time, bisect-angle, copy-paste, scroll-text-2, copy-paste-2,
resize-textarea) and 3 replay tasks (find-word, click-widget, multi-layouts).

Expected finish: 12 h of GRPO plus post-training evals, so about 21:30-22:00
UTC on 2026-11-25, near $37-40. The $42 soft stop ends GRPO early if spend gets
there first. The monitor forces a teardown at $48 or after 20 h.

Success criteria, from `docs/PLAN_RUN4.md`:

- Held-out above run 3's 71.5%.
- Tasks that ran out of turns at 10 turns reach 30% or more.
- At least some drag-task success.
- The full-suite score rises over run 3.
- No importance-ratio blow-ups.

## Where things run

| What | Where |
|---|---|
| Code | GitHub `weights-and-wires/RL-ComputerUse`, branch `main`. Push with the SSH host `git@github-personal`; `gh` user `kashifulhaque`. |
| Secrets | `.env` in the repo root (git-ignored): `RUNPOD_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `HF_ADAPTER_REPO`. Never print the values. |
| Monitor | Bazzite home machine, `ssh -i ~/.ssh/id_ed25519_kh ifkash@192.168.1.4` (Tailscale `100.115.46.42`). Run 4 home: `~/cua-run4`; systemd user unit `cua-run4-monitor`, which acts only while `~/cua-run4/ARMED` exists. Run 3's home `~/cua-run3` is an archive; its unit is disabled. |
| Dashboard | Unit `cua-dashboard` on Bazzite, port 8790. Public at https://run.ifkash.dev: run 4 at `/`, finished runs at `/run3/`, `/run2/`, and `/run1/` (snapshots in `~/cua-runs/`). |
| Public proxy | Caddy in Docker on `vm.ifkash.dev` (`ssh -i ~/.ssh/id_ed25519_kh ifkash@vm.ifkash.dev`). The Caddyfile is `/home/ifkash/docs/infra/caddy/Caddyfile`. The container's bind mount was stale on 2026-11-24, so some host-side edits for `rl.ifkash.dev` aren't live; recreating the container applies them, and that's the owner's call. DNS: Cloudflare A record `run.ifkash.dev`, DNS only. |
| Pod | Runpod, created and deleted by the monitor. The monitor uses Runpod REST v2 (`api.runpod.io/v2`) plus GraphQL for prices and balance; REST v1 (`rest.runpod.io`) no longer exists. SSH into the pod with `~/.ssh/id_ed25519` from the Mac. The pod address is in the monitor status. |
| Checkpoints and results | The Hub repo named by `HF_ADAPTER_REPO`: `checkpoints/grpo-4b-runN/` and `results/runN/`. |
| Notebook site | https://cua-ap9.pages.dev, built by Cloudflare Pages from GitHub `kashifulhaque/cua` (`index.html`, one chapter per run). Run 3's chapter was added on 2026-11-25; run 4's isn't written yet. |

## Check on run 4

All of these are read-only. Run them from the repo root on the Mac.

```bash
scripts/run4_go.sh --status
```

```bash
ssh -i ~/.ssh/id_ed25519_kh ifkash@192.168.1.4 'cd ~/cua-run4 && python3 monitor/monitor.py status && tail -n 10 data/events.jsonl'
```

To read the pod's own log, take the SSH address from the status output:

```bash
ssh -i ~/.ssh/id_ed25519 -p POD_PORT root@POD_IP 'cd /root/RL-ComputerUse && bash scripts/run4.sh --status-json && tail -n 20 results/run4/run.log'
```

Replace `POD_IP` and `POD_PORT` with the values from the monitor status.

Don't stop, restart, or change the run unless the owner asks. To end it
cleanly, the owner runs `scripts/run4_go.sh --stop`, which ends GRPO early,
waits for the evals, uploads, and deletes the pod. `--kill` uploads once and
then deletes the pod right away.

## When run 4 finishes

1. Confirm the teardown in the events: `upload_verified`, then `pod_deleted`.
   Confirm that Runpod lists no pods.
2. Download the results from the Hub into `results/run4/` (`compare/`,
   `eval/*/summary.json`, `eval/*/per_task.csv`, `train/*/metrics.jsonl`,
   `report/`, `run.log`).
3. Report against the success criteria. Check `sampler/drop_capped` and the
   ratio metrics, the drag tasks, and the tasks that used to run out of turns.
4. Add a run 4 chapter to the notebook (`kashifulhaque/cua`), in the style of
   the run 3 chapter: the final story from base to run 4.
5. Disable the `cua-run4-monitor` unit once it reports done.

## Known issues and gotchas

- In shell commands on the Mac, call `/bin/ls`, never `ls` or `eza`. `ls` is
  aliased to `eza --git`, which can hang.
- In zsh, don't keep an SSH command in a variable such as `B='ssh ...'` and
  call `$B`: zsh doesn't split it into words, so it fails.
- Under `set -e`, an assignment whose command substitution exits non-zero ends
  the script. This once stopped `run3_go.sh` before it armed the monitor.
- The Mac's resolver can cache NXDOMAIN for `run.ifkash.dev`. Use
  `curl --resolve run.ifkash.dev:443:167.233.243.236`.
- The built-in browser pane blocks `run.ifkash.dev`. Check the dashboard with
  `curl`, or through Bazzite's LAN address `http://192.168.1.4:8790`.
- Bazzite's `claude` CLI prints a model error line before its output. The
  monitor's note writer drops it.
- The Runpod account bills about $0.008/h with no pods, probably the old run-1
  network volume. It hasn't been deleted; that's the owner's call.
