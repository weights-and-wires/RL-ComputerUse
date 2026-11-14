# Learning guide: RL post-training for computer-use agents

This guide teaches the ideas behind the project in the order you need them. Read
it before Saturday. Each section ends with a short check so you can test
yourself; answers are at the end.

## 1. The problem framed as RL

A computer-use agent is a policy that maps what it sees to an action:

- **State (observation):** the screenshot, the task goal ("Click the button
  labeled OK"), the page's accessibility tree, and the history of actions so far.
- **Action:** one tool call, such as `click("12")` or `fill("7", "Alice")`. For
  a language model, an action is a short sequence of tokens.
- **Episode:** a sequence of turns, from reset until the task ends or the turn
  limit is reached.
- **Reward:** MiniWoB++ checks the page state with JavaScript and returns
  success or failure at the end of the episode. That makes the reward *sparse*
  (one number per episode) and *verifiable* (no human or judge model needed).

The language model is the policy *π_θ*. Its parameters *θ* are what you change.
RL post-training adjusts *θ* so that action sequences leading to reward become
more likely.

**Check 1:** Why does a verifiable reward make RL much easier than training
with a learned reward model?

## 2. Policy gradients in one page

You want to maximize the expected reward *J(θ) = E[R(τ)]* over episodes *τ*
sampled from the policy. The policy-gradient theorem gives:

```text
∇J(θ) = E[ Σ_t ∇ log π_θ(a_t | s_t) · A_t ]
```

In words: for each action taken, push up its log-probability in proportion to
how much better than expected it turned out. *A_t* is the **advantage**: reward
minus a **baseline** (what you expected to get). Subtracting a baseline doesn't
change the expected gradient but cuts its variance enormously.

For an LLM, *log π_θ(a_t | s_t)* is the sum of the log-probabilities of the
tokens in the action. So the update is an ordinary weighted cross-entropy step:
tokens from good episodes get positive weight, tokens from bad episodes get
negative weight.

**Check 2:** If every episode gets reward 1, what is the advantage after you
subtract the mean, and what does the model learn?

## 3. PPO and why GRPO drops the critic

PPO (proximal policy optimization) adds two things:

1. **A critic (value network)** that predicts the expected reward from each
   state, used as the baseline.
2. **A clipped ratio.** Because you reuse a batch of samples for several
   gradient steps, the policy drifts from the one that generated the samples.
   PPO computes `ratio = π_θ(a) / π_old(a)` and clips it to `[1-ε, 1+ε]`, so one
   lucky sample can't move the policy too far.

The critic is a second model as large as the policy, which roughly doubles
memory. **GRPO** (group relative policy optimization, from DeepSeekMath) removes
it: sample a *group* of G episodes for the same task and seed, and use the group
as the baseline:

```text
A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + small)
```

Every token in episode *i* gets advantage *A_i*. Everything else is PPO's
clipped objective. That's the whole algorithm.

Variants used in this project:

- **Clip-higher (DAPO):** a larger upper clip bound (0.28 instead of 0.2) lets
  low-probability good actions grow faster, which keeps exploration alive.
- **No KL penalty (beta = 0):** classic RLHF adds a KL term to stay near the
  reference model. Recent work on verifiable rewards finds it often unnecessary
  and it costs memory, so this project starts without it.
- **Dr. GRPO / token-level normalization:** fixes a length bias where the
  original GRPO loss favors short wrong answers and long right ones.

**Check 3:** With G = 8 and a task the model always fails, what gradient does
GRPO produce? What does this imply for choosing tasks?

## 4. Multi-turn episodes and credit assignment

A single-turn task (a math problem) has one completion per episode. A browser
task has several turns, each with a new observation from the environment. The
project treats the whole episode as one long sequence:

- Assistant tokens (the model's actions) are trained on.
- Environment tokens (screenshots, AXTree text) are **masked out** of the loss.
  The model didn't generate them, so pushing their probability up or down is
  meaningless.
- Every assistant turn in an episode shares the episode's advantage.

This is crude *credit assignment*: if turn 3 was the mistake, turns 1 and 2 are
also penalized. It works because over many episodes the good early actions
appear in successful episodes too. Step-level methods such as GiGPO (in
verl-agent) group identical states across episodes to fix this, at the cost of
complexity. Keep episodes short (8 turns) to limit the damage.

**Check 4:** Why must observation tokens be masked from the loss?

## 5. Why task selection and warm starts matter

GRPO only learns from groups with **reward variance**. Track the fraction of
groups where all G rewards are equal ("zero-variance groups"). If it's above
about 60%, most of your compute is wasted.

Two levers raise the useful fraction:

1. **Task selection:** keep tasks where the base model's sampled success rate
   is between about 10% and 80%.
2. **Rejection-sampling fine-tuning (RFT):** sample many episodes, keep the
   successful ones, and fine-tune on them. This is cheap, needs no teacher, and
   moves tasks from "almost never solved" into the learnable range. It also
   teaches the output format, which removes a whole class of failures.

The 2026 study on GRPO for MiniWoB (arXiv 2607.12640) found no gain on tasks the
model had mastered and up to +22 points on tasks with headroom. That result is
the core design constraint of this project.

## 6. Reward design and reward hacking

The base reward is task success. Shaping terms are small and deliberate:

- A format penalty teaches the model to always output a parseable action.
- An efficiency bonus applies only to successful episodes, so the model can't
  earn it by ending quickly.

*Reward hacking* is when the policy finds a way to score without doing the
task. Typical signs: reward goes up while greedy success on held-out seeds stays
flat; episode length collapses to 1; the model repeats one action. You log each
reward component separately so you can spot this.

**Check 5:** Why would giving the efficiency bonus to failed episodes be a bad
idea?

## 7. The systems side: vLLM, colocation, and LoRA

RL training alternates between two very different workloads:

- **Generation (rollouts):** autoregressive sampling of thousands of short
  completions. A Hugging Face `generate()` loop is far too slow. vLLM uses
  paged attention and continuous batching to make it fast.
- **Training:** forward and backward passes with gradients and optimizer state.

*Colocate mode* runs both on the same GPU: vLLM takes a fixed share of memory
(`vllm_gpu_memory_utilization`), the trainer takes the rest, and after each
optimizer step the new weights are synced into vLLM. On a single GPU, this
avoids paying for a second card that would sit idle half the time.

**LoRA** (low-rank adaptation) freezes the base weights and trains small
low-rank matrices added to each projection layer. Effects in this project:

- Optimizer state is tiny, so a 4B model fits comfortably on one 80 GB card.
- Syncing weights to vLLM means sending only the adapter.
- LoRA tolerates higher learning rates than full fine-tuning (about `1e-5`
  versus about `1e-6`), but RL is still far more fragile than SFT.

The vision tower stays frozen. You're teaching the model *what to do* with what
it sees, not how to see.

**Check 6:** Why does throughput mostly depend on generation speed rather than
the browser?

## 8. Observations: AXTree versus pixels

The main track gives the model both the screenshot and the accessibility tree,
where each element has an ID (`bid`). The model picks an element by ID. That's
much easier than predicting pixel coordinates, so gains show up within a day.

The stretch track (Track B) removes the AXTree and allows only
`mouse_click(x, y)`. That's "true" computer use, the same interface a human or
Claude's computer-use tool uses. Qwen3-VL predicts coordinates on a 0-1000
normalized grid, so the parser rescales them to page pixels. Rewards for this
track can add a grounding signal later, as UI-R1 and GUI-G1 do for single-step
grounding, but the project starts with plain task success.

## 9. How to read the training dashboard

Watch these W&B charts during the run:

| Chart | Healthy | Unhealthy |
|---|---|---|
| Mean reward | Rises slowly, noisy | Flat for 100+ steps, or a sudden drop |
| Zero-variance group fraction | Below 50% and falling | Above 70% |
| Format-error rate | Near zero after RFT | Rising (the policy is degenerating) |
| Entropy | Slowly decreasing | Collapsing to near zero, or spiking |
| Mean turns per episode | Stable or slowly dropping | Collapsing to 1 |
| Greedy eval success (every 50 steps) | Rising | Falling while training reward rises (overfitting or hacking) |

## 10. Suggested reading

Read these in order. The first two are the most important.

1. [DeepSeekMath paper, section 4 (GRPO)](https://arxiv.org/abs/2402.03300)
2. [TRL GRPO trainer documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
3. [GRPO on MiniWoB with Qwen3-VL (arXiv 2607.12640)](https://arxiv.org/abs/2607.12640)
4. [WebAgent-R1: multi-turn RL for web agents](https://arxiv.org/abs/2505.16421)
5. [DAPO: clip-higher and dynamic sampling](https://arxiv.org/abs/2503.14476)
6. [Dr. GRPO: length bias in GRPO](https://arxiv.org/abs/2503.20783)
7. [UI-R1: RL for GUI action prediction](https://arxiv.org/abs/2503.21620)
8. [BrowserGym paper and docs](https://browsergym.readthedocs.io/)
9. [Spinning Up: intro to policy optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

## Exercises before Saturday

1. Read `cua/nano_grpo/loss.py` once the implementation exists. Compute the loss
   by hand for a group of 4 episodes with rewards `[1, 0, 0, 1]`.
2. Open one MiniWoB task in your own browser from the cloned repository and
   solve it. Notice what's hard about it for a model.
3. Predict which 3 candidate tasks will have the most headroom. Compare with the
   baseline results on Saturday.

## Answers

1. A learned reward model can be fooled, and errors in it are optimized against.
   A verifiable reward is exactly the thing you care about.
2. All advantages are 0, so the gradient is 0 and the model learns nothing from
   that group.
3. All rewards are 0, so all advantages are 0: no gradient. Drop tasks the model
   never solves, or warm-start until it sometimes does.
4. The model didn't generate them. Training on them would teach the model to
   predict screenshots and page text, not to act.
5. The model would learn to end episodes quickly, for example by doing nothing
   useful in one turn, and earn a reward for failing faster.
6. Each turn is a VLM forward pass over hundreds of image tokens plus text. A
   MiniWoB step in a browser takes a fraction of a second, and many browsers run
   in parallel on the CPU.
