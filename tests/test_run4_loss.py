"""Run 4's loss and training-loop pieces: the ratio cap, truncated importance
sampling, sampler alignment, per-segment sequences, and the metrics `kind` field.
Needs torch (skipped without it); no GPU.
"""

from __future__ import annotations

import json
import math

import pytest

from cua import pipeline, rollout
from cua.types import EpisodeSpec, SamplingSpec
from tests.test_run4 import FakePool, call, scripted, windowed_cfg

torch = pytest.importorskip("torch")
L = pytest.importorskip("cua.nano_grpo.loss")


def test_ratio_cap_bounds_negative_advantage_tokens_and_zeroes_their_gradient():
    old = torch.zeros(1, 3)
    new = torch.log(torch.tensor([[1.0, 2.0, 1e6]])).requires_grad_()
    adv = torch.tensor([-1.0])
    mask = torch.ones(1, 3)
    loss, m = L.clipped_pg_loss(new, old, adv, mask, 0.2, 0.28, "token_mean", ratio_cap=3.0)
    # Token losses: r * |A| for r <= 3, then 3 * |A|: (1 + 2 + 3) / 3 = 2.
    assert loss.item() == pytest.approx(2.0, rel=1e-5)
    loss.backward()
    assert new.grad[0, 2].item() == 0.0 and new.grad[0, 1].item() != 0.0
    assert m["ratio_capped_frac"] == pytest.approx(1 / 3) and m["ratio_over_cap_frac"] == pytest.approx(1 / 3)
    # Without the cap the same token dominates the loss (run 3, step 207).
    loss2, _ = L.clipped_pg_loss(new.detach(), old, adv, mask, 0.2, 0.28, "token_mean")
    assert loss2.item() > 1e5


def test_ratio_cap_leaves_positive_advantages_and_small_ratios_alone():
    old = torch.zeros(2, 2)
    new = torch.log(torch.tensor([[1.1, 50.0], [0.9, 1.0]]))
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 2)
    a, _ = L.clipped_pg_loss(new, old, adv, mask, 0.2, 0.28, "dr_grpo", 2.0, ratio_cap=3.0)
    b, _ = L.clipped_pg_loss(new, old, adv, mask, 0.2, 0.28, "dr_grpo", 2.0)
    assert a.item() == pytest.approx(b.item())
    with pytest.raises(ValueError):
        L.clipped_pg_loss(new, old, adv, mask, 0.2, 0.28, ratio_cap=1.1)


def test_tis_weights_truncate_and_ignore_unknown_positions():
    trainer = torch.log(torch.tensor([[0.5, 0.9, 0.001, 0.3]]))
    sampler = torch.log(torch.tensor([[0.5, 0.3, 0.5, float("nan")]]))
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    w, m = L.tis_weights(trainer, sampler, mask, cap=2.0)
    assert w[0].tolist() == pytest.approx([1.0, 2.0, 0.002, 1.0], rel=1e-4)
    assert m["tis_capped_frac"] == pytest.approx(0.25)
    # Weights scale the per-token loss.
    adv = torch.tensor([1.0])
    base, _ = L.clipped_pg_loss(trainer, trainer, adv, mask, 0.2, 0.28, "token_mean")
    weighted, _ = L.clipped_pg_loss(trainer, trainer, adv, mask, 0.2, 0.28, "token_mean", is_weights=w)
    assert weighted.item() == pytest.approx(base.item() * w.mean().item(), rel=1e-5)


def test_align_sampler_logprobs_matches_spans_to_turns():
    mask = [0, 1, 1, 1, 0, 0, 1, 1, 0]
    vals, ok = L.align_sampler_logprobs(mask, [[-1, -2, -3], [-4, -5]], ["stop", "stop"])
    assert ok and vals[1:4] == [-1, -2, -3] and vals[6:8] == [-4, -5] and math.isnan(vals[0])
    # A turn cut at max_tokens has no sampled <|im_end|>: its last position stays unknown.
    vals, ok = L.align_sampler_logprobs(mask, [[-1, -2, -3], [-4]], ["stop", "length"])
    assert ok and vals[6] == -4 and math.isnan(vals[7])
    # Re-tokenizing gave another token count: not aligned.
    assert L.align_sampler_logprobs(mask, [[-1, -2], [-4, -5]], ["stop", "stop"])[1] is False
    assert L.align_sampler_logprobs(mask, [[-1, -2, -3]], ["stop"])[1] is False


def test_filter_mismatched_drops_worst_first_within_the_cap():
    diffs = {"a": 0.01, "b": 0.5, "c": 0.2, "d": float("nan"), "e": 0.02, "f": 0.03, "g": 0.01, "h": 0.01}
    aligned = {k: True for k in diffs}
    aligned["d"] = False
    dropped, m = L.filter_mismatched(diffs, aligned, threshold=0.15, max_drop_frac=0.25)
    assert dropped == {"d", "b"}  # unaligned first, then the largest diff; the cap (2 of 8) keeps c
    assert m["sampler/dropped_episodes"] == 2 and m["sampler/drop_capped"] == 1.0
    assert m["sampler/dropped_unaligned"] == 1 and m["sampler/max_abs_logp_diff"] == pytest.approx(0.5)
    dropped, m = L.filter_mismatched(diffs, {k: True for k in diffs}, threshold=None)
    assert dropped == set() and m["sampler/dropped_episodes"] == 0
    assert L.mean_abs_diff([-1.0, -2.0, 0.0], [-1.5, float("nan"), 9.0], [1, 1, 0]) == pytest.approx(0.5)


loop = pytest.importorskip("cua.nano_grpo.loop")


def test_sequence_items_and_episode_minibatches_keep_episodes_whole():
    cfg = windowed_cfg(window=2, max_turns=5)
    eps = rollout.run_episodes([EpisodeSpec("t", s) for s in range(4)], scripted([call("noop")]).generate,
                               FakePool(2), cfg, SamplingSpec(), progress=False)
    items = [(ep, 0.5) for ep in eps]
    seqs = loop.sequence_items(items)
    assert len(seqs) == 4 * 3 and all(conv is ep.conversations()[ci] for ep, _, ci, conv in seqs)
    shards = loop.split_by_episode(seqs, 2)
    assert sum(len(s) for s in shards) == 12
    for shard in shards:
        for ep in {id(x[0]) for x in shard}:
            assert sum(1 for x in shard if id(x[0]) == ep) == 3
    lps, fin = loop.sampler_turns(eps[0], 1)
    assert len(lps) == 2 and fin == ["stop", "stop"]


def test_metrics_rows_carry_a_kind_and_wandb_skips_it():
    row = loop.metrics_row("eval", 25, {"eval/success": 0.5})
    assert row == {"step": 25, "kind": "eval", "eval/success": 0.5}
    assert "kind" not in loop.wandb_values(loop.metrics_row("train", 1, {"loss": 0.1}))
    rows = [loop.metrics_row("train", 1, {}), loop.metrics_row("train", 2, {}), loop.metrics_row("eval", 2, {})]
    assert pipeline.check_metrics(rows) == []
    assert pipeline.check_metrics(rows + [{"step": 3}]) != []
    assert pipeline.check_metrics(rows + [loop.metrics_row("eval", 7, {})]) != []


def test_trim_metrics_keeps_eval_rows_up_to_the_checkpoint(tmp_path):
    path = tmp_path / "m.jsonl"
    rows = [loop.metrics_row("train", 1, {}), loop.metrics_row("train", 2, {}), loop.metrics_row("eval", 2, {}),
            loop.metrics_row("train", 3, {})]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert loop.trim_metrics(path, 2) == 1
    kept = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["step"], r["kind"]) for r in kept] == [(1, "train"), (2, "train"), (2, "eval")]


def test_best_adapter_bookkeeping(tmp_path):
    assert loop.read_best(tmp_path) is None and loop.is_new_best(0.1, None)
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "best.json").write_text(json.dumps({"step": 25, "eval_success": 0.6}))
    best = loop.read_best(tmp_path)
    assert best["step"] == 25 and not loop.is_new_best(0.6, best) and loop.is_new_best(0.61, best)




def _fake_tokenize(processor, messages, tools):
    """Stand-in for policy.tokenize_episode: one token per word, <|im_end|> after each assistant reply."""
    ids, mask = [], []
    for m in messages:
        text = m["content"] if isinstance(m["content"], str) else " ".join(
            b.get("text", "img") for b in m["content"] if isinstance(b, dict))
        words = text.split() or ["_"]
        ids += [1]  # role header
        mask += [0]
        ids += [2 + (sum(map(ord, w)) % 40) for w in words] + ([3] if m["role"] == "assistant" else [])
        mask += [1 if m["role"] == "assistant" else 0] * (len(words) + (m["role"] == "assistant"))
    t = torch.tensor([ids])
    return {"input_ids": t, "attention_mask": torch.ones_like(t), "assistant_mask": torch.tensor([mask])}


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(50, 8)
        self.head = torch.nn.Linear(8, 50)

    def forward(self, input_ids, logits_to_keep=None, **kw):
        logits = self.head(self.emb(input_ids))
        if logits_to_keep is not None:
            logits = logits[:, logits_to_keep]

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_train_phase_runs_segments_with_the_ratio_cap_and_drops_unaligned_episodes(monkeypatch):
    from cua.config import load_config
    from tests.test_run4 import REPO

    monkeypatch.setattr(loop, "tokenize_episode", _fake_tokenize)
    cfg = load_config(REPO / "configs/grpo_run4.yaml", ["env.max_turns=6", "env.context_window_turns=2",
                                                            "grpo.sampler_check.max_mean_abs_diff=100"])
    eps = rollout.run_episodes([EpisodeSpec("t", s) for s in range(8)],
                               scripted([call("noop"), call("double_click", coordinate=[1, 2])]).generate,
                               FakePool(4), windowed_cfg(window=2, max_turns=6), SamplingSpec(), progress=False)
    for ep in eps:
        for t in ep.turns:
            t.generation.logprobs = [-1.0] * (len(t.generation.text.split()) + 1)
            t.generation.finish_reason = "stop"
    eps[0].turns[3].generation.logprobs = [-1.0]  # this one no longer lines up with the trainer's tokens
    model = _TinyLM()
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    from transformers.optimization import get_constant_schedule_with_warmup

    sched = get_constant_schedule_with_warmup(opt, 0)
    items = [(ep, 1.0 if i % 2 else -1.0) for i, ep in enumerate(eps)]
    m = loop.train_phase(model, None, opt, sched, items, cfg, [], micro_bs=2)
    assert m["sampler/dropped_episodes"] == 1 and m["sampler/dropped_unaligned"] == 1
    assert m["train/sequences"] == 7 * 3  # 7 episodes kept, 3 segments each
    for key in ("train/ratio_capped_frac", "train/ratio_over_cap_frac", "train/tis_weight_mean",
                "train/tis_capped_frac", "grad_norm_max", "sampler/mean_abs_logp_diff"):
        assert isinstance(m[key], float) and math.isfinite(m[key]), key
    assert m["sampler/length_mismatch_skips"] == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
