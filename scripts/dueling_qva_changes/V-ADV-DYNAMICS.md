# Dueling DQN: what V and A actually do during training

Measured, not asserted. This file records the value stream, the advantage stream and
the resulting Q during training, under both aggregation rules, plus the gradient
split between the shared trunk and the two heads.

| | paper rule | ablation |
|---|---|---|
| equation | `Q = V + A - mean_a A` | `Q = V + A` |
| config flag | `adv_mean_subtraction = True` | `adv_mean_subtraction = False` |

Everything below comes from `scripts/dueling_instability_probe.py`, which logs
`mean V`, `mean_a A`, `mean_a Q`, `|A|` magnitudes, and the L2 gradient norm of the
trunk and each head every 100-200 steps. Probes are taken on 64 fixed states so the
series are comparable across steps.

**Runs:** CliffWalking-v1, 60k steps, 1.3s/probe. ALE/Breakout-v5, 20k steps,
Apple M4 MPS. Note the much smaller scale than the paper (tens of millions of
frames, DDQN, 512-unit streams, RMSProp).

---

## 1. V and A over training

CliffWalking, 4 actions, 40 sampled points. The value stream does all the work - it
falls toward -22 as the return converges - while the advantage stream stays an order
of magnitude smaller. That is the architecture behaving as intended: most of the
signal is "how bad is this state", not "which action".

### Paper rule - `Q = V + A - mean(A)`

| step | V | mean A | mean Q |
|---|---|---|---|
| 1,200 | -0.13 | -0.010 | -0.13 |
| 7,800 | -12.15 | 1.808 | -12.15 |
| 14,200 | -37.27 | 0.823 | -37.27 |
| 20,800 | -21.54 | 0.390 | -21.54 |
| 27,200 | -21.94 | 0.407 | -21.94 |
| 33,800 | -21.80 | 0.299 | -21.80 |
| 40,200 | -22.07 | 0.395 | -22.07 |
| 46,800 | -21.94 | 0.374 | -21.94 |
| 53,200 | -21.56 | 0.341 | -21.56 |
| 59,800 | -21.75 | 0.386 | -21.75 |

### Plain addition - `Q = V + A`

| step | V | mean A | mean Q |
|---|---|---|---|
| 1,200 | -0.12 | -0.048 | -0.17 |
| 7,800 | -12.60 | -2.396 | -15.00 |
| 14,200 | -20.03 | -3.826 | -23.86 |
| 20,800 | -18.05 | -3.886 | -21.94 |
| 27,200 | -18.10 | -3.844 | -21.95 |
| 33,800 | -17.99 | -3.852 | -21.85 |
| 40,200 | -17.88 | -3.914 | -21.79 |
| 46,800 | -18.06 | -3.976 | -22.04 |
| 53,200 | -17.85 | -3.999 | -21.84 |
| 59,800 | -17.79 | -4.127 | -21.91 |

(10 evenly spaced points out of 294 logged. Every value in these two tables is read
from the log at the step shown.)

**The only difference is the advantage stream.** Under the paper rule it sits just
above zero, pinned by construction. Under plain addition it drifts to about -4.1 and
stays there. V then settles near -17.8 instead of the true -21.9, because it must
absorb the opposite offset.

Read the two tables by column, never by "which line looks bigger". V spans ~38 units
while A spans ~2, so a shared y-axis flattens A into a horizontal stripe and hides
the drift entirely.

---

## 2. V against Q - the identity

This is the clearest single view of what the subtraction buys.

| | paper rule | plain addition |
|---|---|---|
| `V - mean_a Q` | **0.000000** | -4.1265 |
| what that equals | identically zero | `-mean_a A` |
| interpretation | V *is* the state value | V carries an arbitrary constant |

Under the paper rule `V(s) = mean_a Q(s,a)` is forced, so V and Q are not "close" -
they are the *same series*, equal to 4 decimals. Under plain addition they separate
by exactly the unconstrained mean advantage.

Same test on Breakout: paper-rule curves stay welded together; plain curves fan apart
as the advantage stream grows (gap reaches ~0.22 mean, up to 0.39 in `|A|` terms).

---

## 3. The advantage stream grows - nothing diverges

No blow-up, no NaN. Plain addition simply lets `|A|` expand. Breakout, `mean|A|`
across 64 probe states:

| step | `mean|A|` paper | `mean|A|` plain |
|---|---|---|
| 1,100 | 0.035 | 0.088 |
| 3,200 | 0.040 | 0.164 |
| 5,300 | 0.032 | 0.200 |
| 7,400 | 0.035 | 0.220 |
| 9,500 | 0.032 | 0.327 |
| 11,500 | 0.031 | 0.370 |
| 13,600 | 0.036 | 0.262 |
| 15,700 | 0.035 | 0.201 |
| 17,800 | 0.035 | 0.228 |
| 19,900 | 0.032 | 0.222 |

The constrained net is flat at ~0.035 for the entire run. The unconstrained one
climbs to 0.370 before easing back - a 6.8x ratio at the mean.

---

## 4. Gradient norm by stream: Q, V and A

Three norms per step:

- `grad_total` - all online parameters (trunk + both heads). This is the "Q" stream.
- `grad_values` - only the value head's parameters.
- `grad_adv` - only the advantage head's parameters.

**These are parameter-gradient norms, not gradients of the V and A outputs.** The
total is therefore *not* `sqrt(V^2 + A^2)`: the trunk is a third block.

The three parameter sets are disjoint, so `||g_total||^2 = ||g_trunk||^2 + ||g_v||^2
+ ||g_a||^2` exactly (no cross terms). Verified numerically: zero negative residuals
across all 294 + 189 logged steps.

### CliffWalking

| step | total (paper) | V (paper) | A (paper) | total (plain) | V (plain) | A (plain) |
|---|---|---|---|---|---|---|
| 1,200 | 27.2 | 24.5 | 10.1 | 31.7 | 26.5 | 15.9 |
| 7,800 | 134.1 | 67.4 | 70.4 | 83.8 | 31.3 | 39.9 |
| 14,200 | 13.3 | 5.4 | 4.3 | 12.9 | 2.4 | 4.7 |
| 20,800 | 6.4 | 3.8 | 1.5 | 6.5 | 4.4 | 2.8 |
| 27,200 | 1.4 | 0.1 | 0.5 | 1.5 | 0.5 | 1.0 |
| 33,800 | 1.8 | 0.7 | 0.9 | 2.9 | 2.0 | 1.7 |
| 40,200 | 1.6 | 1.0 | 0.8 | 2.6 | 0.7 | 2.2 |
| 46,800 | 4.2 | 2.8 | 2.5 | 6.0 | 4.2 | 4.0 |
| 53,200 | 6.1 | 4.1 | 3.7 | 1.5 | 0.7 | 1.2 |
| 59,800 | 3.3 | 0.7 | 2.2 | 2.9 | 1.4 | 2.4 |

Full-series means:

| stream | paper | plain | change |
|---|---|---|---|
| value head | 18.3 | 21.0 | +15% |
| advantage head | 12.5 | 18.2 | **+46%** |
| total | 31.6 | 37.5 | +18% |
| peak total | 478 | 722 | 1.51x |

### The transient is not one spike

Both runs fire repeatedly across the first ~9k steps. At full resolution (294 logged
points) the peak is 478 / 722; the downsampled rows below are representative, not
exhaustive:

| step | paper | plain |
|---|---|---|
| 2,800 | 434 | 722 |
| 4,200 | 225 | 242 |
| 5,800 | 97 | 121 |
| 7,200 | **245** | **232** |
| 8,800 | 149 | 160 |

After the transient the norm is small but not flat:

| window | paper max | paper median | plain max | plain median |
|---|---|---|---|---|
| step > 8,000 | 216.3 | 3.53 | 307.9 | 2.77 |
| step > 10,000 | 35.4 | 3.40 | **93.3** | 2.63 |
| step > 15,000 | 31.7 | 3.18 | 23.0 | 2.51 |
| step > 20,000 | 11.3 | 3.15 | 23.0 | 2.53 |

Two things to take from this. First, there is still a 307.9 spike between steps 8k and
10k, so "done by 9k" is approximate. Second, the early-window maxima are *unstable as
a statistic*: sampling 40 evenly spaced points reports a post-10k maximum of 59.4 for
the plain run, while the full log reports 93.3 at step 11,400. Any claim built on a
downsampled maximum should be read with that in mind.

The medians, which are far more robust, tell the same story throughout: 3.4 vs 2.6
after step 10k, and both settle to ~3.

Any chart of this needs two scales: full range (0-750) to see the transient, and a
zoom (0-60) to see the settled region. A single capped axis at 70 hides the 245 spike;
a single full-range axis makes everything after step 10k invisible.

### Breakout

| stream | paper | plain | ratio |
|---|---|---|---|
| `mean|A|` | 0.0362 | 0.2447 | **6.8x** |
| `max|A|` | 0.0483 | 0.2793 | 5.8x |
| advantage-head grad | 0.0269 | 0.0616 | **2.3x** |
| value-head grad | 0.0868 | 0.1093 | 1.3x |
| total grad, mean | 0.282 | 0.581 | 2.1x |
| total grad, peak | 1.29 | 3.92 | **3.0x** |
| `max|V|` | 0.977 | 0.784 | 0.80x |
| TD loss | 0.0068 | 0.0075 | 1.1x |

---

## 5. Where the gradient energy goes

Absolute norms do not compare across environments (CliffWalking runs at 30+,
Breakout at 0.3). Normalized share of `||g||^2` by parameter block, averaged over all
logged steps:

| run | shared trunk | value head | advantage head |
|---|---|---|---|
| CliffWalking - paper | 42.5% | 43.1% | **14.3%** |
| CliffWalking - plain | 37.7% | 41.3% | **21.0%** |
| Breakout - paper | 90.8% | 8.5% | 0.7% |
| Breakout - plain | 96.6% | 2.8% | 0.7% |

Each column sums to 100%.

**CliffWalking is the clean case.** Dropping the subtraction moves the advantage head
from 14.3% to 21.0% of all gradient energy - a 47% relative increase - while the value
head barely budges. The advantage stream is precisely what absorbs the unconstrained
drift: `|A|` floats, so `dL/dtheta_a` grows, and the optimiser keeps taking bigger
steps in the branch that no longer has to mean anything.

**Breakout resists that reading.** The trunk holds ~91% of the energy, so the heads
are noise in share terms even when the advantage head doubles - it sits at 0.7% in
both columns. The absolute advantage gradient grows 2.3x but the distribution of
energy barely moves; the trunk swelling 2.1x is what drives the total. The
CliffWalking attribution does not transfer here.

---

## 6. Is the offset real? Same task, three seeds

This is the test that has nothing to do with transients. Three seeds, same config,
same 60k steps. All six runs solve CliffWalking: every eval return is **-13.0**, the
optimum.

| seed | eval return | V (paper) | V (plain) | Q (paper) | Q (plain) | V-Q gap (plain) |
|---|---|---|---|---|---|---|
| 0 | -13.0 | -21.66 | -20.76 | -21.66 | -21.70 | **0.95** |
| 1 | -13.0 | -21.28 | -17.88 | -21.28 | -21.31 | **3.43** |
| 2 | -13.0 | -22.54 | -19.28 | -22.54 | -22.51 | **3.23** |
| spread | 0.0 | 1.26 | 2.88 | 1.26 | 1.20 | 2.48 |

If the V/A split were pinned down by the task, solving the task would pin it down. It
does not. Q agrees across seeds (spread 1.20) and the return is identical to the
decimal, yet under plain addition V lands somewhere different every time. The gap is
set by the random seed, not by the task.

That is the fingerprint of a flat direction: the loss is exactly flat along the gauge
direction, so nothing pushes back, the noise wanders, and wherever it stops it stays.
It is not a transient and not a divergence - it is a permanent arbitrary offset.

Under the paper rule the gap is identically 0 at every seed.

---

## 7. Why identifiability, and not the rest of the network

The intuition is available by reading the equation, before any training run.

The network outputs two free functions: `V(s)` from one head, `A(s,a)` from the
other. The definition of advantage is `A(s,a) = Q(s,a) - V(s)`. Nothing in the
architecture enforces that - the advantage head is a generic function approximator and
will output whatever minimises the loss. So the natural question at design time is:
*"I named an output 'advantage'; does my network actually compute the advantage?"*

With mean subtraction the constraint is satisfied identically. For `Q = V + A -
mean_a A` to hold with V depending only on s, you get `V(s) = mean_a Q(s,a)` forced.
Substituting back:

```
A(s,a) = Q(s,a) - mean_a Q(s,a) = Q(s,a) - V(s)
```

which is exactly `Q - V`. The subtraction is not a stability hack - it is the
constraint that makes the head compute the quantity it is named after. Without it,
`A = Q - V` for an arbitrary `V`, so the head's output is not the advantage at all.

Why no equivalent concern for other parts of the network? Neural nets are full of
unidentifiability - hidden units can be permuted, ReLU layers rescaled `(aW, b/a)`
with identical output - and nobody worries, because those units make no claim about
what they mean. V and A are different in exactly one way: they are given names and
sold as interpretations. A named quantity owes you identifiability in a way a feature
map does not.

---

## 8. What the paper actually says

Wang et al. (2016) are explicit that equation (7), `Q = V + A`, is **unidentifiable**,
and assert that "lack of identifiability is mirrored by poor practical performance
when this equation is used directly."

What they publish is the arithmetic plus that assertion. What they do **not** publish
is any ablation or diagnostic comparing the two rules on Atari - no gradient-norm
curve, no V/A magnitude trace, no divergence figure, in the paper or its appendix.

So the identifiability argument stands on its own (it is a two-line proof), while the
practical-instability claim is not documented. That matters when interpreting our
numbers: they are not "what the researchers saw", they are what the two streams do
when someone actually watches them.

Their reported reasoning around the fix: first try forcing `max_a A = 0`, which is
also a valid gauge - but it changes only one action's advantage per update, so it is
jumpy. Mean subtraction is the smoother compromise. Both are gauge choices, which is
itself telling: they were thinking in terms of gauge freedom from the start.

---

## 9. Final losses, and why they do not settle the question

End-of-run values for everything. Both variants use the same seed (1000) and identical
data order within each environment, so this is as fair a comparison as the setup
allows.

| metric | CliffWalking paper | CliffWalking plain | Breakout paper | Breakout plain |
|---|---|---|---|---|
| Final TD loss | 0.01250 | **0.00358** | 0.00392 | **0.00174** |
| Last-10% mean loss | 0.01110 | **0.00590** | **0.00333** | 0.00478 |
| Overall mean loss | 50.30 | 57.58 | 0.00681 | 0.00751 |
| Final mean V | -21.75 | -17.79 | 0.911 | 0.737 |
| Final mean A | 0.386 | -4.127 | 0.032 | 0.222 |
| Final `|V - Q|` | **0.000** | 4.127 | **0.000** | 0.222 |
| Last-20 episode return | -30.05 | -30.05 | 1.05 | 2.10 |

Episode returns above are from training (epsilon >= 0.05). The greedy eval return is
-13.0 for both on CliffWalking.

### The lower loss is on the variant without the fix

Plain addition ends with the *lower* TD loss in both environments - 0.00358 against
0.01250 on CliffWalking, 0.00174 against 0.00392 on Breakout - while its V has drifted
4.13 away from Q.

This is not a contradiction, it is the point. The gauge direction is flat in the loss,
so the extra free parameter costs nothing and the network is free to spend it on
fitting. **The loss cannot see the thing that went wrong.** A run can reach a lower TD
loss while the quantity you named "state value" has stopped being one.

### Two caveats that stop this being over-read

1. **"Final" is a single noisy sample.** On Breakout the last-10% mean reverses the
   ordering (0.00333 paper vs 0.00478 plain), so the final-step number alone should
   not carry weight.
2. **The overall means run the other way** (50.30 vs 57.58 on CliffWalking), because
   the plain run's early transient is larger.

Where the loss is informative at all, it is in the tail - and there the picture is
mixed:

| after step 10k, loss > 0.25 | values |
|---|---|
| paper rule | 10.2k=0.70, 11.8k=0.66, **13.2k=3.19**, **14.8k=7.01**, 16.2k=2.07, 20.8k=1.16, 29.8k=0.56, 38.8k=0.28 |
| plain | **10.2k=23.48**, 11.8k=0.39, 14.8k=0.62, 20.8k=1.25 |

Neither variant is stable in the tail, and they fail differently. Plain posts the
single largest post-10k spike (23.48 at step 10.2k) - larger than anything the
constrained run produces after its transient - but is then comparatively quiet, with
only three more excursions above 0.25. The paper rule produces no single spike that
large but keeps firing: eight excursions, three of them above 2.0.

So steady-state TD loss does not cleanly favour either rule on this task. That is
worth stating plainly rather than hiding behind the gradient numbers.

### Reading the loss charts

CliffWalking TD loss spans 856 down to 0.003 - over five orders of magnitude - so it
needs two scales: full range (0-1000) for the transient, and a zoom to 0-2 for the
tail (clipping 9/40 paper and 7/40 plain points, all early). Breakout is tight enough
for a single axis, but both series are noisier step-to-step than the gap between them.

---

## 10. What this does not show

- **The correlation only holds on Breakout.** There, `corr(mean|A|, grad norm)` is
  0.59 for plain against -0.01 for constrained. On CliffWalking both sit near -0.57,
  so gradient size is not explained by A's drift in that run. One environment is not
  a mechanism.
- **Plain addition still converged.** Both variants reached the optimal -13 return on
  CliffWalking. The drifting V did not stop learning at this scale. The claim being
  supported is about what V *means*, not that training fails.
- **Different scale from the paper.** 60k and 20k steps, not tens of millions of
  frames; vanilla DQN, not DDQN; 64-unit streams, not 512; Adam, not RMSProp.
- **Clipping hides the spikes.** Every run clips at 0.5; the paper clips at 10. The
  ratios above are lower bounds.
- **V's early plunge is not the gauge.** On CliffWalking V sweeps from ~0 to ~-22 in
  the first ~10k steps as returns are learned. That is ordinary convergence, present
  in both variants. Only the *separation* between V and Q is the gauge effect.
- **No log-scale chart.** These series span ~2.5 orders of magnitude within a single
  run, so any single linear axis either hides the transient or hides the tail. The
  two-scale presentation in section 4 is a workaround, not a solution.

---

## Reproducing

```bash
# CliffWalking, with per-step instrumentation
python scripts/dueling_instability_probe.py --steps 60000 --log-every 200

# Atari (needs: uv pip install "gymnasium[atari]" ale-py)
python scripts/dueling_instability_probe.py \
    --env-id ALE/Breakout-v5 --steps 20000 --log-every 100

# Cross-seed gauge test
python scripts/dueling_gauge_proof.py --seeds 0 1 2
```

Source artifacts: `runs/dueling_instability_cliff/instability.json`,
`runs/dueling_instability_atari/instability.json`,
`runs/dueling_qva_seed{0,1,2}.log`.
