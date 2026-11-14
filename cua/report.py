"""Build the HTML report: `python -m cua.report`.

Reads what the other modules wrote under `results/` and produces one page,
`results/report/index.html`:

- headline numbers: base vs RFT vs GRPO (greedy success) on the train and held-out pools
- per-task grouped bar charts, one per pool
- training curves from `results/train/*/metrics.jsonl`
- GIF pairs: base vs GRPO on the same task and seed
- a failure-mode table for every eval run

Charts are PNGs embedded as base64, so the HTML needs no other files except
the GIFs, which are copied to `results/report/gifs/` and referenced by relative
path (embedding many GIFs would make the page huge).

Missing runs and files never raise errors. The page shows "not run" instead.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from cua.config import add_config_args, config_from_args, results_dir

GOAL_POINTS = 15.0  # PLAN.md: train-pool success must rise by at least 15 points

# Eval run names, fixed by the "Output conventions" in the module contract.
POOLS: dict[str, dict[str, str]] = {
    "train": {"base": "base-train", "rft": "rft-train", "grpo": "grpo-train"},
    "heldout": {"base": "base-heldout", "rft": "rft-heldout", "grpo": "grpo-heldout"},
}
POOL_TITLES = {"train": "Training pool", "heldout": "Held-out pool"}
LABELS = {"base": "Base", "rft": "RFT", "grpo": "GRPO"}
# Colors for the bars. They stay readable on the white chart background.
COLORS = {"base": "#8a8f98", "rft": "#e69f00", "grpo": "#0072b2"}
FAILURE_MODES = ["success", "wrong_action", "invalid_action", "format_error", "ran_out_of_turns", "crashed"]
FAILURE_LABELS = {
    "success": "Success",
    "wrong_action": "Wrong action",
    "invalid_action": "Invalid action",
    "format_error": "Format error",
    "ran_out_of_turns": "Ran out of turns",
    "crashed": "Browser crash",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_summaries(eval_dir: Path) -> dict[str, Optional[dict]]:
    """Return {run_name: summary or None} for every run the report knows about."""
    names = {n for pool in POOLS.values() for n in pool.values()}
    return {name: _read_json(eval_dir / name / "summary.json") for name in sorted(names)}


def load_episodes(eval_dir: Path, name: str) -> list[dict]:
    path = eval_dir / name / "episodes.jsonl"
    rows: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def load_metrics(train_dir: Path) -> dict[str, list[dict]]:
    """Return {run_name: metric rows} for each `results/train/<run>/metrics.jsonl`."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(train_dir.glob("*/metrics.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a half-written last line while training still runs
        if rows:
            runs[path.parent.name] = rows
    return runs


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def esc(x: Any) -> str:
    return html.escape(str(x))


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def png_b64(fig: Any) -> str:
    """Serialize a matplotlib figure to a data URI and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img_tag(data_uri: str, alt: str) -> str:
    # The chart has a white background in both themes (see .chart in CSS), so its
    # text stays readable when the page is dark.
    return f'<img class="chart" src="{data_uri}" alt="{esc(alt)}">'


def not_run(what: str) -> str:
    return f'<p class="muted">Not run: {esc(what)}</p>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def headline_section(summaries: dict[str, Optional[dict]]) -> str:
    """Cards with success rates and deltas against the base model."""
    parts = []
    for pool, runs in POOLS.items():
        base = summaries.get(runs["base"])
        base_rate = base.get("overall_success") if base else None
        cards = []
        for key in ("base", "rft", "grpo"):
            s = summaries.get(runs[key])
            rate = s.get("overall_success") if s else None
            if s is None or rate is None:
                cards.append(
                    f'<div class="card"><div class="label">{LABELS[key]}</div>'
                    f'<div class="big muted">not run</div><div class="small">{esc(runs[key])}</div></div>'
                )
                continue
            delta_html = ""
            if key != "base" and base_rate is not None:
                delta = 100 * (rate - base_rate)
                cls = "good" if delta > 0 else ("bad" if delta < 0 else "muted")
                delta_html = f'<div class="delta {cls}">{delta:+.1f} points vs base</div>'
                if pool == "train":
                    verdict = "goal met" if delta >= GOAL_POINTS else f"below the +{GOAL_POINTS:.0f} goal"
                    delta_html += f'<div class="small">{verdict}</div>'
                else:
                    delta_html += f'<div class="small">{"no drop" if delta >= 0 else "dropped"}</div>'
            cards.append(
                f'<div class="card"><div class="label">{LABELS[key]}</div>'
                f'<div class="big">{pct(rate)}</div>{delta_html}'
                f'<div class="small">{s.get("n_episodes", "?")} episodes, {s.get("num_seeds", "?")} seeds</div></div>'
            )
        parts.append(f"<h3>{POOL_TITLES[pool]}</h3><div class=\"cards\">{''.join(cards)}</div>")
    note = (
        f"<p class=\"small muted\">Greedy decoding on held-out seeds. Goal: at least +{GOAL_POINTS:.0f} points "
        "on the training pool, and no drop on the held-out pool.</p>"
    )
    return "<h2>Headline numbers</h2>" + note + "".join(parts)


def per_task_chart(pool: str, summaries: dict[str, Optional[dict]]) -> str:
    """Grouped bars: one group per task, one bar per model."""
    import matplotlib

    matplotlib.use("Agg")  # no display on the pod
    import matplotlib.pyplot as plt

    present = {k: summaries[n] for k, n in POOLS[pool].items() if summaries.get(n)}
    if not present:
        return not_run(f"{POOL_TITLES[pool]} evals ({', '.join(POOLS[pool].values())})")
    tasks = sorted({t for s in present.values() for t in s.get("per_task", {})})
    if not tasks:
        return not_run("no per-task data")

    width = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(tasks) + 2), 4.2))
    for i, (key, s) in enumerate(present.items()):
        vals = [(s["per_task"].get(t) or {}).get("success_rate") for t in tasks]
        xs = [j + (i - (len(present) - 1) / 2) * width for j in range(len(tasks))]
        ax.bar(xs, [100 * (v or 0) for v in vals], width, label=LABELS[key], color=COLORS[key])
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Greedy success (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{POOL_TITLES[pool]}: success per task")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return img_tag(png_b64(fig), f"Per-task success, {POOL_TITLES[pool]}")


def _series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for i, row in enumerate(rows):
        if key in row and isinstance(row[key], (int, float)):
            xs.append(row.get("step", i))
            ys.append(row[key])
    return xs, ys


def _find_key(all_rows: dict[str, list[dict]], exact: list[str], fuzzy: Optional[re.Pattern]) -> Optional[str]:
    """First key from `exact` that any run logged, else the first key matching `fuzzy`."""
    keys = {k for rows in all_rows.values() for row in rows for k in row}
    for k in exact:
        if k in keys:
            return k
    if fuzzy:
        for k in sorted(keys):
            if fuzzy.search(k):
                return k
    return None


def _smooth(ys: list[float], window: int = 5) -> list[float]:
    out = []
    for i in range(len(ys)):
        chunk = ys[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def curves_section(metrics: dict[str, list[dict]], eval_dir: Path) -> str:
    if not metrics:
        return "<h2>Training curves</h2>" + not_run("no results/train/*/metrics.jsonl files")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Total reward", _find_key(metrics, ["reward/total"], None)),
        ("Success rate", _find_key(metrics, ["success/overall"], None)),
        ("Zero-variance groups (fraction)", _find_key(metrics, ["groups/zero_variance_frac"], None)),
        (
            "Format errors",
            _find_key(
                metrics,
                ["format_error_rate", "format/error_rate", "errors/format_rate"],
                re.compile(r"format.*(err|rate)|(err).*format", re.I),
            ),
        ),
        ("Policy entropy", _find_key(metrics, ["entropy"], re.compile(r"entropy", re.I))),
        ("Gradient norm", _find_key(metrics, ["grad_norm"], re.compile(r"grad_?norm", re.I))),
    ]
    shown = [(title, key) for title, key in panels if key]
    parts = ["<h2>Training curves</h2>"]
    if shown:
        cols = 2
        rows_n = (len(shown) + 1) // 2
        fig, axes = plt.subplots(rows_n, cols, figsize=(10, 3 * rows_n), squeeze=False)
        for ax in axes.flat[len(shown):]:
            ax.axis("off")  # hide unused cells
        for ax, (title, key) in zip(axes.flat, shown):
            for run, rows in metrics.items():
                xs, ys = _series(rows, key)
                if not xs:
                    continue
                (line,) = ax.plot(xs, ys, alpha=0.3, linewidth=1)
                if len(ys) >= 5:  # a moving average makes noisy RL curves readable
                    ax.plot(xs, _smooth(ys), color=line.get_color(), linewidth=2, label=run)
                else:
                    line.set_alpha(1.0)
                    line.set_label(run)
            ax.set_title(f"{title}  ({key})", fontsize=9)
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        parts.append(img_tag(png_b64(fig), "Training curves"))
        missing = [t for t, k in panels if not k]
        if missing:
            parts.append(f'<p class="small muted">Not logged: {esc(", ".join(missing))}.</p>')
    else:
        parts.append(not_run("metrics files hold none of the expected keys"))

    # Mid-training greedy evals (grpo-step<N>-train) form a coarse success-vs-step curve.
    mid = []
    for path in eval_dir.glob("grpo-step*-train/summary.json"):
        m = re.search(r"grpo-step(\d+)-train", str(path.parent.name))
        s = _read_json(path)
        if m and s and s.get("overall_success") is not None:
            mid.append((int(m.group(1)), s["overall_success"]))
    if mid:
        mid.sort()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([m[0] for m in mid], [100 * m[1] for m in mid], marker="o", color=COLORS["grpo"])
        ax.set_xlabel("training step")
        ax.set_ylabel("Greedy success (%)")
        ax.set_title("Mid-training eval (training pool)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        parts.append(img_tag(png_b64(fig), "Mid-training greedy success"))
    return "".join(parts)


def gif_pairs_section(eval_dir: Path, out_dir: Path, max_pairs: int = 24) -> str:
    """Grid of base-vs-GRPO GIFs on identical task and seed."""
    parts = ["<h2>Replays: base vs GRPO</h2>"]
    pairs: list[tuple[int, str, str, str, Optional[bool], Optional[bool]]] = []
    for pool in POOLS:
        base_run, grpo_run = POOLS[pool]["base"], POOLS[pool]["grpo"]
        base_dir, grpo_dir = eval_dir / base_run / "gifs", eval_dir / grpo_run / "gifs"
        if not (base_dir.is_dir() and grpo_dir.is_dir()):
            continue
        outcome = {
            run: {(e["task"], e["seed"]): e["success"] for e in load_episodes(eval_dir, run) if e.get("sample_idx", 0) == 0}
            for run in (base_run, grpo_run)
        }
        for gif in sorted(base_dir.glob("*.gif")):
            if not (grpo_dir / gif.name).exists():
                continue
            m = re.match(r"(.+)_seed(\d+)\.gif$", gif.name)
            key = (m.group(1), int(m.group(2))) if m else None
            b, g = outcome[base_run].get(key), outcome[grpo_run].get(key)
            # Show pairs where GRPO fixed a failure first: they teach the most.
            priority = 0 if (b is False and g is True) else (1 if b != g else 2)
            pairs.append((priority, pool, gif.name, key[0] if key else gif.stem, b, g))
    if not pairs:
        return "".join(parts) + not_run("need GIFs from both base-* and grpo-* eval runs on the same seeds")

    pairs.sort(key=lambda p: (p[0], p[1], p[2]))
    cells = []
    for _, pool, name, task, b, g in pairs[:max_pairs]:
        sources = {}
        for label, run in (("base", POOLS[pool]["base"]), ("grpo", POOLS[pool]["grpo"])):
            dest = out_dir / "gifs" / run / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(eval_dir / run / "gifs" / name, dest)
            sources[label] = f"gifs/{run}/{name}"

        def badge(ok: Optional[bool]) -> str:
            return "unknown" if ok is None else ("success" if ok else "fail")

        cells.append(
            f'<figure class="pair"><figcaption><b>{esc(name[:-4])}</b> <span class="small muted">({esc(POOL_TITLES[pool])})</span></figcaption>'
            f'<div class="pairgrid">'
            f'<div><img loading="lazy" src="{esc(sources["base"])}" alt="Base replay of {esc(name)}"><div class="small">Base: {badge(b)}</div></div>'
            f'<div><img loading="lazy" src="{esc(sources["grpo"])}" alt="GRPO replay of {esc(name)}"><div class="small">GRPO: {badge(g)}</div></div>'
            f"</div></figure>"
        )
    parts.append(f'<div class="pairs">{"".join(cells)}</div>')
    if len(pairs) > max_pairs:
        parts.append(f'<p class="small muted">Showing {max_pairs} of {len(pairs)} pairs. All GIFs stay in results/eval/*/gifs/.</p>')
    return "".join(parts)


def failure_section(summaries: dict[str, Optional[dict]]) -> str:
    parts = ["<h2>Failure modes</h2>"]
    rows = []
    for pool, runs in POOLS.items():
        for key, name in runs.items():
            s = summaries.get(name)
            if not s or not s.get("failure_modes"):
                rows.append(f'<tr><td>{esc(name)}</td><td colspan="{len(FAILURE_MODES)}" class="muted">not run</td></tr>')
                continue
            modes = s["failure_modes"]
            total = sum(modes.values()) or 1
            cells = "".join(
                f"<td>{modes.get(m, 0)} <span class=\"small muted\">({100 * modes.get(m, 0) / total:.0f}%)</span></td>"
                for m in FAILURE_MODES
            )
            rows.append(f"<tr><td>{esc(name)}</td>{cells}</tr>")
    head = "".join(f"<th>{esc(FAILURE_LABELS[m])}</th>" for m in FAILURE_MODES)
    parts.append(f"<div class=\"scroll\"><table><thead><tr><th>Run</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    parts.append(
        '<p class="small muted">Counts are episodes. "Format error" means the model produced no valid tool call at least once. '
        '"Wrong action" means valid actions that did not solve the task.</p>'
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root { --bg:#ffffff; --fg:#1d1f23; --muted:#6b7280; --card:#f4f5f7; --line:#dfe2e7; --good:#15803d; --bad:#b91c1c; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14161a; --fg:#e6e8eb; --muted:#9aa1ab; --card:#1e2127; --line:#32363e; --good:#4ade80; --bad:#f87171; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { margin-bottom: 4px; } h2 { margin-top: 40px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
.muted { color: var(--muted); } .small { font-size: 13px; }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.label { color:var(--muted); font-size:13px; } .big { font-size:32px; font-weight:650; }
.delta { font-weight:600; } .good { color:var(--good); } .bad { color:var(--bad); }
.chart { max-width:100%; height:auto; background:#fff; border-radius:8px; padding:6px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
.pairs { display:grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap:16px; }
.pair { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; }
.pairgrid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:6px; }
.pairgrid img { width:100%; height:auto; border-radius:6px; }
@media (max-width: 480px) { .pairs { grid-template-columns: 1fr; } }
"""


def build_report(cfg: Any, out_dir: Optional[Path] = None) -> Path:
    root = results_dir(cfg)
    eval_dir, train_dir = root / "eval", root / "train"
    out_dir = out_dir or root / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(eval_dir)
    body = [
        headline_section(summaries),
        "<h2>Success per task</h2>",
        *[f"<h3>{POOL_TITLES[p]}</h3>{per_task_chart(p, summaries)}" for p in POOLS],
        curves_section(load_metrics(train_dir), eval_dir),
        gif_pairs_section(eval_dir, out_dir),
        failure_section(summaries),
    ]
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>RL for computer use: report</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>RL post-training Qwen3-VL for browser tasks</h1>"
        '<p class="muted">MiniWoB++ via BrowserGym. Base vs rejection-sampling fine-tuning (RFT) vs GRPO.</p>'
        + "".join(body)
        + "</main></body></html>"
    )
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build results/report/index.html from eval and training outputs.")
    add_config_args(parser, "configs/eval.yaml")
    parser.add_argument("--out", default=None, help="Output directory. Default: results/report.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    path = build_report(cfg, Path(args.out) if args.out else None)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
