# Proof: why Dueling DQN needs the mean-advantage subtraction

**Question.** Does the advantage head, and specifically the `mean(A)` subtraction,
actually do anything — or is it decoration?

**Answer.** It does nothing to the policy and nothing to `Q`. It does everything
to `V` and `A`: it is the only thing that makes them *mean* anything. Without it,
`V(s)` is not the state value — it is the state value plus an arbitrary
constant.

That is not an opinion or a training-curve impression. It is arithmetic, and both
parts below are measured on trained networks.

Reproduce with `scripts/dueling_gauge_proof.py`.

---

## 1. The arithmetic

### Why plain addition is ambiguous

```
Q = V + A
```

Take any constant `c` and replace `V → V + c`, `A → A - c`. Then

```
(V + c) + (A - c) = V + A = Q
```

`Q` is **bit-identical**. Since DQN's loss is computed on `Q` only, nothing in
training can distinguish `(V, A)` from `(V + c, A - c)`. The pair is
*unidentifiable* — there is a whole family of decompositions of the same `Q`, and
the network arbitrarily lands on one.

### Why the mean subtraction is unambiguous

```
Q = V + A - mean_a A
```

Shift again:

```
(V + c) + (A - c) - mean_a(A - c)
  = V + c + A - c - mean_a A + c
  = Q + c
```

`Q` moves by exactly `c`. The symmetry is **destroyed**, so only one
representative satisfies the rule. That representative is the one with
`mean_a A = 0`, and it implies

```
mean_a Q = V + mean_a A - mean_a A = V
```

So the subtraction is *equivalent* to the statement **`V(s) = mean_a Q(s,a)`**.
That is the entire purpose: it turns `V` into the actual state value.

### Worked example, by hand

One real state from a trained CliffWalking network (4 actions), `c = 5`:

| | `V` | `A` (4 actions) | `mean_a A` |
|---|---|---|---|
| before | -27.1543 | +15.2052, -84.9903, +14.1585, +14.2181 | -10.3521 |
| after `V+5`, `A-5` | -22.1543 | +10.2052, -89.9903, +9.1585, +9.2181 | -15.3521 |

Now combine:

| rule | `Q` before | `Q` after | change |
|---|---|---|---|
| `Q = V + A` | -11.949148 | -11.949148 | **0.000000** |
| `Q = V + A - mean(A)` | -1.597004 | +3.402996 | **+5.000000** |

Plain addition cannot see the shift. The mean rule sees it exactly.

And here is what that does to `V` on that same state: the network reports
`V = -27.15`, but the four `Q` values are `-11.95, -112.14, -13.00, -12.94`, whose
mean is `-37.50`. So under plain addition `V` is off by `10.35` — exactly
`mean_a A`. Under the mean rule that gap is forced to zero.

---

## 2. Experiment A — the gauge shift, measured

Shift `V → V + c`, `A → A - c` on a trained net, recompute `Q` under each rule,
and record the largest change over all states and actions.

**CliffWalking-v1** (60k steps, converged):

| `c` | `Q = V + A` max abs ΔQ | `Q = V + A - mean(A)` max abs ΔQ |
|---|---|---|
| -10.0 | 1.91e-06 | **10.000** |
| -5.0 | 1.91e-06 | **5.000** |
| -1.0 | 9.54e-07 | **1.000** |
| 0.0 | 0.00e+00 | 0.000 |
| +1.0 | 9.54e-07 | **1.000** |
| +5.0 | 9.54e-07 | **5.000** |
| +10.0 | 9.54e-07 | **10.000** |

**Atari Breakout, ALE/Breakout-v5** (40k steps): the same pattern, with plain
addition at `2.4e-07` to `9.5e-07` and the mean rule again exactly `|c|`.

The `~1e-6` on the plain row is float32 rounding, not signal — it is the residual
of `(V+c) + (A-c)` computed in 32-bit. The mean row is `|c|` to machine
precision, i.e. the shift is fully visible.

Read it this way: **the plain network is invariant to a change that the mean
network detects exactly.** A quantity the loss cannot see is a quantity the loss
cannot pin down.

### The shift is policy-irrelevant

The paper states that subtracting the mean "does not change the relative rank of
the `A` (and hence `Q`) values, preserving any greedy or ε-greedy policy". This
is true and we confirm it: for every `c` tested, the greedy action
`argmax_a Q(s,a)` was preserved under **both** rules.

This is the key point for the doubt you started with. The gauge freedom is
**pure redundancy** — it cannot change any decision. So constraining it costs
nothing in performance. It buys interpretability for free.

---

## 3. Experiment B — is `V` actually the state value?

From `Q = V + A_eff` it follows that

```
V - mean_a Q = -mean_a A_eff
```

so `V = mean_a Q` holds **exactly** when `mean_a A_eff = 0`. Measuring
`V - mean_a Q` on trained networks therefore tests the theorem directly, with no
confound from training dynamics.

**CliffWalking-v1** — 60k steps, 3 seeds, 1000 shared states:

| variant | seed | mean abs(`V - mean_a Q`) | max abs | identity `V - mean_a Q = -mean_a A_eff` |
|---|---|---|---|---|
| `Q = V + A - mean(A)` | 1000 | **2.00e-07** | 3.82e-06 | holds |
| | 1001 | **2.22e-07** | 3.82e-06 | holds |
| | 1002 | **1.07e-06** | 3.82e-06 | holds |
| `Q = V + A` | 1000 | **4.334** | 10.85 | holds |
| | 1001 | **3.112** | 7.63 | holds |
| | 1002 | **7.409** | 19.67 | holds |

**Atari Breakout** — 40k steps, 2 seeds, 300 shared states:

| variant | seed | mean abs(`V - mean_a Q`) | max abs |
|---|---|---|---|
| `Q = V + A - mean(A)` | 1000 | **2.42e-08** | 2.38e-07 |
| | 1001 | **2.08e-08** | 1.19e-07 |
| `Q = V + A` | 1000 | **0.307** | 0.487 |
| | 1001 | **0.132** | 0.187 |

Seven orders of magnitude apart on Atari, four on CliffWalking. Under the mean
rule, `V` **is** the average action value to float precision. Under plain
addition it is not, and the error equals the unconstrained mean advantage —
confirmed exactly in the last column of the CliffWalking table.

`02_v_is_state_value.png` shows this visually: scatter `V` against `mean_a Q`.
The mean-subtracted net sits exactly on the diagonal; the plain net scatters off
it by an arbitrary amount.

---

## 4. The alleged practical instability, measured

The paper says unidentifiability is "mirrored by poor practical performance when
this equation is used directly" — but publishes no diagnostic. No gradient-norm
plot, no V/A magnitude curve. So here is what the numbers do when instrumented
(`scripts/dueling_instability_probe.py`, logging every 100–200 steps).

**Nothing diverges.** There is no blow-up, no NaN. What happens is subtler: the
advantage stream *grows*.

Breakout, mean `|A(s,a)|` over a fixed probe set:

| steps | `Q = V + A - mean(A)` | `Q = V + A` |
|---|---|---|
| 1.1k | 0.035 | 0.088 |
| 8.3k | 0.036 | 0.300 |
| 11.2k | 0.030 | 0.372 |
| 19.9k | 0.032 | 0.222 |

The constrained net sits flat around `0.036` for the entire run. The
unconstrained net climbs to `~0.37` — about **6.8x** larger — before easing back.

Gradients follow it:

| metric (Breakout) | `V + A - mean(A)` | `V + A` | ratio |
|---|---|---|---|
| grad norm, mean | 0.282 | 0.581 | 2.1x |
| grad norm, p99 | 1.22 | 3.29 | 2.7x |
| grad norm, peak | 1.29 | 3.92 | **3.0x** |
| TD loss | 0.0068 | 0.0075 | 1.1x |

CliffWalking shows the same direction, weaker: grad peaks `478` vs `722`
(1.5x), mean `31.6` vs `37.5` (1.2x), and `mean_a A` drifting from `-0.05` to
`-4.13` while the constrained version stays near zero.

### Where this evidence is weak

- **The correlation only holds on Breakout.** There,
  `corr(mean|A|, grad norm)` is `0.59` for the plain rule against `-0.01` for the
  constrained one. On CliffWalking both are around `-0.57`, so drift does *not*
  explain gradient size there. One environment is not a mechanism.
- **Plain addition did not fail.** Both variants reached the optimal `-13` on
  CliffWalking. Bigger gradients and a drifting advantage stream did not stop
  convergence at this scale.
- **Clipping hides the spikes.** Every run clips at `0.5`; the paper clips at
  `10`. So the reported ratios are lower bounds on the raw difference.

Reading: the mechanism is real and measurable, but at this scale it is a whisper,
not a failure. The paper's stronger claim about practical performance remains
their assertion.

---

## 5. So what does the advantage head do?

**The two-stream architecture** is the performance idea. In most states the action
barely matters; `V` is learned once and shared across actions instead of being
relearned per action. This is what the paper's gains come from (`Duel` beats
`Single`, more so as redundant actions are added).

**The mean subtraction** is not a performance idea. It is what makes the shared
`V` a *value* rather than a value plus a free constant. It changes the meaning of
the two heads; it does not change `Q` or the policy at all.

That distinction answers the original puzzle directly: the ablation was never
supposed to change results. It was supposed to — and did — change what `V` and
`A` denote.

---

## 6. What the paper actually says, and what it does not

To be precise about the provenance, from Wang et al. (2016):

- They call equation (7), `Q = V + A`, **unidentifiable**, and state that this
  "lack of identifiability is mirrored by poor practical performance when this
  equation is used directly." That is the claim this document verifies.
- They address it by forcing zero advantage at the chosen action, then note it
  "loses the original semantics of `V` and `A` because they are now off-target by
  a constant, but ... increases the stability of the optimization." All their
  reported experiments use the mean form, equation (9).
- Their Atari evidence is **Duel vs Single-stream**, across 57 games, plus an
  ablation on prioritized replay. They report experimenting with a softmax
  variant of the aggregation, and finding it similar.
- **The paper does not publish a per-game Atari table comparing
  `Q = V + A - mean(A)` against `Q = V + A`.** The "poor practical performance"
  sentence is asserted, not tabulated.

So: the gauge argument and the `V = mean_a Q` identity above are settled
mathematics, and we confirmed both on trained networks. The claim that the
unconstrained form *performs worse at Atari scale* is the paper's assertion, and
this work does **not** independently confirm it — our plain-addition agents
trained fine.

## 7. What was not replicated

Explicitly, so the evidence is not over-claimed:

- **Not 57 games.** Two environments: CliffWalking-v1 and ALE/Breakout-v5.
- **Not 50M frames / 200M frames.** 60k steps (CliffWalking) and 40k steps
  (Breakout) — small by Atari standards. Both Breakout agents are far from
  human-level; the value *identity* does not depend on convergence, which is why
  it is still a valid test, but the *returns* here are not a performance result.
- **Not the paper's DDQN + prioritized replay setup**, and not their network
  sizes, optimizer settings, or frame-skip regime. This is NeatRL's `DuelingQNet`
  and its own hyperparameters.
- **No Human Starts metric**, no 30-no-op evaluation protocol.

What *is* established: the identifiability argument, exactly, with numbers, on
both a small converged task and the paper's Atari benchmark.

---

## 8. Reproducing

```bash
# Converged small task, 3 seeds, fast (~7 min)
.venv/bin/python scripts/dueling_gauge_proof.py \
  --env-id CliffWalking-v1 --steps 60000 --seeds 3 --states 1000 \
  --device mps --out-dir runs/dueling_gauge_proof_cliff

# Atari Breakout, 2 seeds (~13 min on M4)
.venv/bin/python scripts/dueling_gauge_proof.py \
  --env-id ALE/Breakout-v5 --steps 40000 --seeds 2 --states 300 \
  --buffer-size 10000 --device mps --out-dir runs/dueling_gauge_proof_atari

# Re-analyse without retraining
.venv/bin/python scripts/dueling_gauge_proof.py \
  --env-id CliffWalking-v1 --steps 60000 --seeds 3 --states 1000 \
  --out-dir runs/dueling_gauge_proof_cliff --reuse
```

Atari needs `uv pip install "gymnasium[atari]" ale-py`.

Each run writes:

| File | Contents |
|---|---|
| `01_gauge_shift.png` | ΔQ versus shift `c`, both rules |
| `02_v_is_state_value.png` | `V` against `mean_a Q`, both variants |
| `proof_metrics.json` | every number above, plus the hand-checkable state |
| `per_seed.npz` | raw `Q`/`V`/`A` per seed, for `--reuse` |
