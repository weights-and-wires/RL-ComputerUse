"""The GRPO math: group advantages, token log-probabilities, and the clipped loss.

Everything here is a pure function of tensors. There is no I/O, no model, and
no global state, so you can read each function on its own and test it on tiny
inputs.

Three steps turn a batch of episodes into a loss:

1. `group_advantages` turns the rewards of the G episodes that share a (task,
   seed) into advantages: how much better than its group average was each one?
2. `token_logprobs` turns the model's logits into log pi(token | prefix) for the
   tokens that were actually sampled.
3. `clipped_pg_loss` combines them into one scalar. Tokens from
   better-than-average episodes get their probability pushed up. Tokens from
   worse-than-average episodes get pushed down.

Worked example: rewards [1, 0, 0, 1] (exercise 1 in docs/LEARNING.md)
=====================================================================

Setup: one group of G = 4 episodes. Episodes 1 and 4 succeed, episodes 2 and 3
fail. Each episode has a different number of assistant tokens, because the model
writes more text in some episodes than in others:

    episode   reward   assistant tokens
    1         1        20
    2         0        40
    3         0        30
    4         1        10

Step 1, advantages. The group mean is (1 + 0 + 0 + 1) / 4 = 0.5. The group
standard deviation uses the unbiased estimator that `torch.std` computes by
default: sqrt(((0.5)^2 * 4) / (4 - 1)) = sqrt(1/3) = 0.5774. So:

    A = (r - 0.5) / (0.5774 + 1e-4)  ->  [+0.866, -0.866, -0.866, +0.866]

The advantages always sum to zero within a group. The two winners share +0.866
and the two losers share -0.866. Every token of an episode gets that episode's
advantage, including tokens of turns that did nothing wrong. That is the crude
credit assignment described in docs/LEARNING.md section 4.

Step 2, the ratio. On the first optimizer step after sampling, the policy that
you train is identical to the policy that sampled, so ratio = pi_new / pi_old =
1 for every token and clipping never triggers. The per-token loss is then just
-A * ratio = -A:

    episode 1: -0.866 on each of 20 tokens    (sum = -17.32)
    episode 2: +0.866 on each of 40 tokens    (sum = +34.64)
    episode 3: +0.866 on each of 30 tokens    (sum = +25.98)
    episode 4: -0.866 on each of 10 tokens    (sum =  -8.66)

Step 3, reduce to one number. The three `loss_type` variants differ only in
the divisor. Let B = 4 be the number of episodes and C = 40 be a constant token
budget (the `max_tokens_norm` argument):

    grpo        mean over each episode's tokens, then mean over episodes
                (-0.866 + 0.866 + 0.866 - 0.866) / 4                    = 0.000
    token_mean  total sum / total tokens = 34.64 / 100                  = 0.346
    dr_grpo     total sum / (B * C)      = 34.64 / 160                  = 0.217

The loss value is not what trains the model. The gradient is. With ratio = 1,
the gradient of the loss with respect to each token's log-probability is
-A / divisor. A negative number means that raising the token's log-probability
lowers the loss, so the optimizer pushes that token up. Here is that weight
per token, for each variant:

    episode        grpo (-A / (B * len))   token_mean (-A / 100)   dr_grpo (-A / 160)
    1 (A=+0.866)   -0.01083                -0.00866                -0.00541
    2 (A=-0.866)   +0.00541                +0.00866                +0.00541
    3 (A=-0.866)   +0.00722                +0.00866                +0.00541
    4 (A=+0.866)   -0.02165                -0.00866                -0.00541

Read down the columns to see why the variants exist:

- `grpo` divides by each episode's own length. A token in the short winner
  (episode 4, 10 tokens) gets twice the push of a token in the long winner
  (episode 1, 20 tokens), and a token in the long loser (episode 2, 40 tokens)
  gets a quarter of that push. Long wrong answers are punished less per token
  than short wrong answers, which is the length bias that Dr. GRPO describes.
- `token_mean` (the DAPO choice) gives every token the same weight regardless
  of which episode it sits in. Long episodes contribute more total gradient
  because they contain more tokens.
- `dr_grpo` also gives every token the same weight, but it divides by a
  constant that does not depend on the batch. The loss scale then stays stable
  from step to step, and micro-batches add up exactly.

Check your work: the loss is the sum of weight x token count over the
episodes. For `token_mean`, that is 0.00866 * (-20 + 40 + 30 - 10) = 0.346. For
`dr_grpo`, it is 0.00541 * (-20 + 40 + 30 - 10) = 0.217. Both match the loss
values above.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LOSS_TYPES = ("dr_grpo", "token_mean", "grpo")


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4, scale: bool = True) -> torch.Tensor:
    """Computes the advantage of each episode relative to its own group.

    GRPO needs no value network (critic). The other episodes that share the same
    task and seed are the baseline: an episode is "good" when it beats the
    group average.

    Args:
      rewards: Float tensor of shape `[num_groups, G]`. Row i holds the total
        rewards of the G episodes sampled for prompt i. G must be at least 2,
        because the standard deviation of one value is undefined.
      eps: Added to the standard deviation so a group where every reward is
        equal produces advantages of exactly 0 instead of 0/0.
      scale: If True, divide by the group standard deviation, which is the
        original GRPO recipe. If False, only subtract the mean, which is the
        Dr. GRPO recipe. Dividing by a small standard deviation inflates the
        weight of groups that are almost solved or almost hopeless.

    Returns:
      Tensor with the same shape as `rewards`. Each row sums to zero.
    """
    rewards = rewards.float()
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    if scale:
        advantages = advantages / (rewards.std(dim=1, keepdim=True) + eps)
    return advantages


def _gather_logp(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log_softmax(logits)[ids] for one chunk, without keeping a full log-softmax."""
    logits = logits.float()
    # log p(id) = logit[id] - logsumexp(logits). This avoids materializing the
    # normalized `[B, chunk, vocab]` log-softmax tensor; only the sum is needed.
    chosen = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
    """Returns log pi(input_ids[t] | input_ids[:t]) for every position t.

    A causal language model's logits at position t predict the token at
    position t + 1. So the log-probability of `input_ids[:, t]` comes from
    `logits[:, t - 1]`. Position 0 has no prefix, so its value is 0 and you must
    keep it masked out (the assistant mask never covers it).

    Memory matters here. Qwen3-VL has a vocabulary of about 151,000 tokens. A
    single 8,000-token episode in fp32 would need 8,000 x 151,000 x 4 bytes =
    4.8 GB for the log-softmax alone, and autograd would save more copies. Two
    tricks keep the peak small:

    - The function processes `chunk_size` positions at a time, so the fp32
      copy of the logits is only `chunk_size x vocab`.
    - When gradients are on, each chunk runs under activation checkpointing:
      autograd stores only the (bf16) logits slice and recomputes the fp32 math
      during the backward pass. Without checkpointing, chunking would not
      reduce the saved memory, because autograd would keep every chunk's fp32
      intermediate until backward.

    Args:
      logits: Tensor of shape `[B, T, V]`, typically bf16.
      input_ids: Integer tensor of shape `[B, T]`.
      chunk_size: Positions per chunk. Lower it if you still run out of memory.

    Returns:
      Float32 tensor of shape `[B, T]`, aligned with `input_ids`.
    """
    use_ckpt = torch.is_grad_enabled() and logits.requires_grad
    pieces = []
    num_predicted = logits.shape[1] - 1  # the last logit predicts a token we don't have
    for start in range(0, num_predicted, chunk_size):
        end = min(start + chunk_size, num_predicted)
        chunk_logits = logits[:, start:end]
        chunk_ids = input_ids[:, start + 1 : end + 1]
        if use_ckpt:
            pieces.append(checkpoint(_gather_logp, chunk_logits, chunk_ids, use_reentrant=False))
        else:
            pieces.append(_gather_logp(chunk_logits, chunk_ids))
    logp = torch.cat(pieces, dim=1)
    return F.pad(logp, (1, 0))  # prepend the dummy value for position 0


def logit_positions(mask: torch.Tensor) -> torch.Tensor:
    """Returns the positions whose logits predict a masked (assistant) token.

    The logits at position t predict the token at t + 1, so a token at position
    t needs the logits at t - 1. The result covers the union over the batch,
    which is what `logits_to_keep` (a 1D index tensor, the same for every
    sample) accepts. For one episode per micro-batch it is exact; for more it
    adds a few positions that the mask zeroes out later.

    Args:
      mask: `[B, T]`, 1 on assistant tokens.

    Returns:
      Sorted 1D long tensor of positions in `[0, T - 2]`, on the mask's device.
    """
    tokens = torch.nonzero(mask.bool().any(dim=0), as_tuple=False).flatten()
    tokens = tokens[tokens >= 1]  # position 0 has no prefix and is never an assistant token
    return tokens - 1


def token_logprobs_at(
    logits: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor, chunk_size: int = 1024
) -> torch.Tensor:
    """`token_logprobs`, from logits computed only at `positions`.

    Use this with a model called as `model(..., logits_to_keep=positions)`,
    which applies the vocabulary projection only at those positions. The
    returned tensor has the same `[B, T]` layout as `token_logprobs`, with the
    log-prob of `input_ids[:, p + 1]` at index `p + 1` and 0 everywhere else,
    so the masks and slicing downstream stay unchanged.

    Args:
      logits: `[B, K, V]`, where `K = len(positions)`.
      input_ids: `[B, T]`.
      positions: 1D long tensor from `logit_positions`.
      chunk_size: Positions per chunk, as in `token_logprobs`.

    Returns:
      Float32 tensor of shape `[B, T]`.
    """
    targets = input_ids[:, positions + 1]
    use_ckpt = torch.is_grad_enabled() and logits.requires_grad
    pieces = []
    for start in range(0, logits.shape[1], chunk_size):
        chunk_logits = logits[:, start : start + chunk_size]
        chunk_ids = targets[:, start : start + chunk_size]
        if use_ckpt:
            pieces.append(checkpoint(_gather_logp, chunk_logits, chunk_ids, use_reentrant=False))
        else:
            pieces.append(_gather_logp(chunk_logits, chunk_ids))
    logp = torch.cat(pieces, dim=1) if pieces else logits.new_zeros(input_ids.shape[0], 0, dtype=torch.float32)
    out = torch.zeros(input_ids.shape, dtype=torch.float32, device=logits.device)
    out[:, positions + 1] = logp  # index assignment keeps the autograd graph
    return out


def microbatch_weight(
    loss_type: str, micro_tokens: float, total_tokens: float, micro_seqs: int, total_seqs: int
) -> float:
    """Returns the factor that makes gradient accumulation match one big batch.

    `clipped_pg_loss` normalizes over the micro-batch it receives. When you
    split an optimizer step into several micro-batches, multiply each
    micro-batch loss by this weight before calling `backward()`. The summed
    result then equals the loss over the whole step:

    - `token_mean` divides by the number of tokens, so weight by the token share.
    - `grpo` and `dr_grpo` divide by quantities that scale with the number of
      episodes, so weight by the episode share.
    """
    if loss_type == "token_mean":
        return micro_tokens / max(total_tokens, 1.0)
    return micro_seqs / max(total_seqs, 1)


def clipped_pg_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps_low: float,
    eps_high: float,
    loss_type: str = "dr_grpo",
    max_tokens_norm: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the PPO-style clipped policy-gradient loss, token by token.

    For each token the ratio is `r = pi_new(token) / pi_old(token)`. The loss
    takes the more pessimistic of two estimates, which removes any incentive to
    move `r` outside `[1 - eps_low, 1 + eps_high]`:

        per_token_loss = -min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A)

    Setting `eps_high` above `eps_low` is "clip-higher" (DAPO). It lets a
    low-probability token with a positive advantage grow more before clipping
    stops it, which helps the policy keep exploring.

    The three `loss_type` options differ only in how the per-token losses
    (zero outside `mask`) are reduced to a scalar:

    - `"grpo"`: the original recipe. Average over each episode's tokens, then
      average over episodes. Every episode has equal weight, so each token in a
      long episode counts for less. This biases training toward short wrong
      answers and long right ones.
    - `"token_mean"`: the DAPO recipe. Sum over all tokens in the batch and
      divide by the token count. Every token has equal weight.
    - `"dr_grpo"`: the Dr. GRPO recipe. Sum over tokens and divide by the
      constant `num_episodes * max_tokens_norm`. Every token has equal weight
      and the scale does not depend on how long this batch happens to be.

    See the module docstring for a numeric comparison.

    Args:
      new_logp: `[B, T]` log-probs from the model that you are training
        (carries gradients).
      old_logp: `[B, T]` log-probs from the policy that generated the data,
        detached. When the data is exactly on-policy, pass `new_logp.detach()`.
      advantages: `[B]` one advantage per episode.
      mask: `[B, T]` 1 on assistant tokens, 0 elsewhere (observations, padding).
      eps_low: Lower clip distance, for example 0.2.
      eps_high: Upper clip distance, for example 0.28.
      loss_type: One of `LOSS_TYPES`.
      max_tokens_norm: Constant divisor for `dr_grpo`: the largest number of
        assistant tokens an episode can have. If None, the longest sequence in
        this batch is used, which makes the scale batch-dependent.

    Returns:
      A tuple `(loss, metrics)`. `loss` is a scalar tensor normalized over the
      batch you passed in. `metrics` holds plain floats: `clip_frac` (share of
      tokens where clipping changed the result), `approx_kl` (the k3 estimator
      of KL(old || new)), `ratio_mean`, `ratio_max`, and `entropy_proxy`
      (-mean of the sampled-token log-prob, which tracks entropy because the
      tokens were sampled from the policy).
    """
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
    mask = mask.float()
    adv = advantages.float().unsqueeze(1)  # [B, 1] broadcasts over tokens

    # log-ratio, with masked positions zeroed so padding can never produce inf.
    log_ratio = (new_logp - old_logp) * mask
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    clipped_ratio = ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    unclipped_obj = ratio * adv
    clipped_obj = clipped_ratio * adv
    per_token = -torch.min(unclipped_obj, clipped_obj) * mask

    num_seqs = per_token.shape[0]
    seq_tokens = mask.sum(dim=1)
    if loss_type == "grpo":
        loss = (per_token.sum(dim=1) / seq_tokens.clamp(min=1)).mean()
    elif loss_type == "token_mean":
        loss = per_token.sum() / mask.sum().clamp(min=1)
    else:  # dr_grpo
        norm = max_tokens_norm if max_tokens_norm is not None else seq_tokens.max().clamp(min=1)
        loss = per_token.sum() / (num_seqs * norm)

    with torch.no_grad():
        n = mask.sum().clamp(min=1)
        # Clipping is "active" when the min() picked the clipped branch.
        clipped = (clipped_obj < unclipped_obj).float() * mask
        # k3 estimator for samples drawn from old: r - 1 - log r. It is >= 0
        # and has lower variance than the naive log-ratio.
        k3 = (ratio - 1.0 - log_ratio) * mask
        metrics = {
            "clip_frac": (clipped.sum() / n).item(),
            "approx_kl": (k3.sum() / n).item(),
            "ratio_mean": (ratio * mask).sum().div(n).item(),
            "ratio_max": (ratio * mask).max().item(),
            "entropy_proxy": (-(new_logp * mask).sum() / n).item(),
        }
    return loss, metrics
