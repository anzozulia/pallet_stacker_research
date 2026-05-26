# BR1 SOTA gap — analysis + proposed experiments

Current state (post-port, n=10, 30 s budget):

| System | BR1 mean util₁ |
|---|---:|
| Gonçalves-Resende 2013 | 92.62 % |
| v3.12 + Cython port | 91.03 % |
| **Gap** | **−1.59 pp** |

The port gave throughput but not algorithm-side gains on BR1: every
instance ties exactly with v3.12 baseline (0/10/0 W/T/L on BR1, vs
+0.21 / +0.33 / +0.46 pp on BR3/5/7). Conclusion: **BR1 saturates** —
extra generations don't help, so the gap is structural, not budgetary.

This doc records what's known about the saturation, audits the polish
stack, and proposes ranked experiments to close the gap.

---

## 1. Baseline-strength check

The v2 PalletPacker alone (no BRKGA) on the first 3 BR1 instances:

| Instance | N | v2-alone util₁ | BRKGA-final util₁ | BRKGA contribution |
|---|---:|---:|---:|---:|
| BR1#1 | 112 | 79.32 % | 91.05 % | +11.73 pp |
| BR1#2 | 138 | 88.04 % | 91.36 % | +3.32 pp |
| BR1#3 | 127 | 73.20 % | 88.11 % | +14.91 pp |

BRKGA is doing real work — it's not just "v2 already gets us 91 %." The
question is what's stopping the final +1.6 pp.

---

## 2. Polish stack audit

`pallet_packer/_brkga_core/polish.py` provides three operators:

### 2.1 `local_search_2opt` (4 s budget, ON by default)

Hill-climbing on BPS positions:

| Move | Frequency | Description |
|---|---:|---|
| Position swap | 30 % | swap boxes at two random positions |
| Segment reverse | 25 % | reverse k=2..7 consecutive positions |
| Insert | 20 % | move one box from i to j |
| Rotation flip | 20 % | randomize one box's rotation key |
| Decoder flip | 5 % | randomize the selector key |

**Acceptance**: strict improvement only (no SA, no plateau-walk).

Concern: hill-climbing from one starting point with only local moves is
classically stuck in narrow basins. Empirically BR1 hits the same util
across runs at varying budgets → likely stuck in one such basin.

### 2.2 `path_relinking` (max 100 evals, ON by default)

Walks from chrom_a (best) → chrom_b (second-best) by progressively
copying chrom_b's keys into chrom_a. Top-2 only.

Concern: top-2 path is narrow. With pop_size=600 across K=3 populations
the elites span >10 distinct local optima — top-2 PR ignores them.

### 2.3 `lns_polish` (variable budget, **OFF by default**)

Destroys K random keys (5–25 % of n), re-decodes, accepts if better.
Designed exactly to escape the basin that 2-opt can't reach.

**This is OFF in the canonical config.** Re-enabling is one of the
first experiments to try.

### 2.4 What's missing

Compared to BR-tuned literature implementations:

- **Block-aware LS moves**: swap entire same-SKU blocks between
  positions. Currently swap moves are box-level.
- **Multi-start LS**: LS from top-K elites, not just the global best.
- **Reactive search**: adapt move probabilities based on success.
- **Simulated annealing-style acceptance**: small uphill moves can
  escape narrow basins.
- **Per-instance polish budget**: easy instances finish in 5 s and
  waste the remaining 25 s on saturated BRKGA; LS could absorb that.

---

## 3. Proposed experiments — ranked by expected gain × effort

| # | Experiment | Expected gain | Effort | Risk |
|---|---|---:|---|---|
| 1 | Enable LNS by default | 0.2–0.5 pp | < 1 h | low |
| 2 | Parallel LS (Phase 5b batch) | 0.3–0.8 pp | 0.5 d | low |
| 3 | LS from top-K elites in parallel | 0.4–0.7 pp | 0.5 d | low |
| 4 | Multi-pair path relinking (top-5) | 0.2–0.4 pp | 0.3 d | low |
| 5 | Block-aware LS swap operator | 0.3–0.8 pp | 1 d | medium |
| 6 | SA-style acceptance in LS | 0.2–0.5 pp | 0.5 d | medium |
| 7 | Multi-start BRKGA (n_restarts=2-3) | −0.1 to +0.5 pp | < 1 h | low |
| 8 | Reactive operator probabilities | 0.1–0.3 pp | 0.5 d | medium |

Estimated **realistic combined gain** with all of 1–6: 1.0–2.5 pp.
That would close the SOTA gap with margin.

### 3.1 Why parallel LS is high-leverage

Current LS at 4 s budget runs ~1300–4000 serial decodes (depends on
mode mix). With the Phase 5 batch infrastructure, we can evaluate
N=80 candidate moves at once → 8 × more LS iterations in the same
budget → many more chances to escape the basin. **This is the single
biggest infrastructure leverage we haven't applied yet.**

Implementation sketch:
```python
while time.time() - t0 < budget:
    candidates = generate_batch(80, current_chrom, rng)
    fits = decode_population_fitness(candidates, ...)  # one parallel call
    best_idx = np.argmin(fits)
    if fits[best_idx] < best_fit - 1e-9:
        current = candidates[best_idx]; best_fit = fits[best_idx]
```

### 3.2 Why multi-start LS is high-leverage

Currently LS runs on the single global best. With parallel batch we
can run LS from the top-K elites simultaneously (each as a separate
batch path). If different elites land in different basins, LS from
each gets a different chance.

---

## 4. Diagnostic results

### 4.1 Saturation curve (BR1#8 at 60/120/240 s)

| Budget | Final util | Actual runtime |
|---|---:|---:|
| 60 s | 94.33 % | 49.0 s |
| 120 s | 94.33 % | 50.1 s |
| 240 s | 94.33 % | 49.8 s |

**The algorithm terminates at ~49 s regardless of available budget**
and produces the identical 94.33 % util. This is `patience=200` kicking
in — with Phase 5's faster decode rate, 200 stagnant generations now
take ~10–20 s instead of the original ~80 s.

The full 240 s budget run only uses 49 s. We're leaving **80 % of the
budget unused** when stuck.

### 4.2 LNS A/B (5 instances, 30 s budget)

| Instance | LS only | LS + LNS | Δ |
|---|---:|---:|---:|
| BR1#1 | 91.05 % | 91.05 % | +0.00 |
| BR1#2 | 91.36 % | 92.20 % | **+0.84** |
| BR1#3 | 88.11 % | 88.11 % | +0.00 |
| BR1#4 | 86.91 % | 86.91 % | +0.00 |
| BR1#5 | 94.61 % | 94.63 % | +0.02 |
| **Mean** | 90.41 % | 90.58 % | **+0.17** |

LNS alone is marginal — BR1#2 is the only meaningful gain. The
hypothesis is that LNS at the *end* (after BRKGA already terminated)
isn't enough — the BRKGA itself stopped at saturation and the LNS
gets a single short pass from the local optimum.

### 4.3 Implications

The real bottleneck is **early termination via patience**, not the
absence of LNS. The right fix is one of:

- **Bump patience** (e.g. 200 → 2000): keeps BRKGA running through
  the full budget so polish + diversification mechanisms have time.
- **Patience-triggered LNS** instead of termination: when patience
  fires, perturb elites + continue.
- **Patience-triggered restart**: full reset of non-elite half.

A focused test is currently running (`test_patience_impact.py`) to
quantify the effect of `patience=200` vs `patience=2000` ± LNS.

### 4.4 Patience test results — and the diagnosis pivot

| Instance | p=200 | p=2000 | p=2000+LNS | rt p=200 | rt p=2000 |
|---|---:|---:|---:|---:|---:|
| BR1#1 | 91.05 % | 91.05 % | 91.05 % | 15.8 s | 30.0 s |
| BR1#2 | 91.36 % | 91.36 % | **92.20 %** | 30.1 s | 30.1 s |
| BR1#3 | 88.11 % | 88.11 % | 88.11 % | 17.2 s | 30.0 s |
| BR1#4 | 86.91 % | 86.91 % | 86.91 % | 30.1 s | 30.1 s |
| BR1#5 | 94.61 % | 94.63 % | 94.63 % | 23.4 s | 30.0 s |

**Patience alone unlocks 0 instances.** Even when forced to use the full
30 s budget (e.g. BR1#1 going from 15.8 s → 30 s), util is identical.
**The local optimum is sticky** — more BRKGA compute on the same
population doesn't escape it.

LNS unlocks 1 instance (BR1#2 by +0.84 pp); doesn't move the others.

### 4.5 Revised diagnosis — basin attraction, not BRKGA compute

The 4/5 instances stuck at the same value are evidence that BRKGA is
attracted into a basin (likely shaped by the v2 seed + smart init) and
cannot escape with the current operators. The 1/5 that responds to LNS
suggests the basin has a "thin wall" sometimes, traversable by random
key destruction.

This shifts the highest-leverage fix from "more BRKGA time" to:

- **Multi-start with different seeds**: 30 s total, split into 2×15 s
  or 3×10 s with different seeds. Different basins explored.
- **Skip v2 seed**: the v2 PalletPacker seed might be pulling BRKGA
  toward the same basin every time. Removing it might find a different
  one.
- **Patience-triggered LNS** (not termination): when patience fires,
  destroy 50 % of non-elite keys and continue. Already partially tested
  via LNS post-BRKGA; running LNS *inside* BRKGA might compound.
- **Different decoder mix**: literature uses layer-based decoders
  exclusively for BR; we use 1/6 layer + 5/6 other. Biasing mode_cdf
  toward layer modes might find better basins.

These are now the priority experiments.

### 4.6 Diversification test results

5 BR1 instances × 5 configs at 30 s budget each:

| Instance | base | r=2 | r=3 | no-v2 | r2+LNS | best Δ |
|---|---:|---:|---:|---:|---:|---:|
| BR1#1 | 91.05 | **92.02** | **92.02** | 91.18 | **92.02** | +0.97 |
| BR1#2 | 91.36 | 91.36 | 91.36 | **92.62** | 91.36 | **+1.26** |
| BR1#3 | 88.11 | 88.11 | 88.11 | 88.11 | 88.11 | +0.00 |
| BR1#4 | 86.91 | 86.91 | 86.91 | 86.83 | 86.91 | +0.00 |
| BR1#5 | 94.61 | 94.63 | 94.63 | 94.63 | 94.63 | +0.02 |

Per-config mean across 5 BR1 instances:
- base: 90.41 %
- r=2 : 90.61 % (+0.19)
- r=3 : 90.61 % (+0.19)
- no-v2: 90.67 % (+0.27)
- r2+LNS: 90.61 % (+0.19)
- **best-per-instance** (oracle): 90.86 % (**+0.45**)

### 4.7 Conclusions

1. **No single config dominates.** BR1#1 wants `n_restarts=2`; BR1#2
   wants `no v2 seed`. Different instances live in different basins.

2. **BR1#3 and BR1#4 are hard-stuck.** None of the diversification
   strategies help. Likely needs block-aware LS, different decoder
   modes, or has a true structural ceiling.

3. **Multi-strategy within a single run is the right next step.** Use
   the existing K=3 multi-population to assign different seeding
   strategies per population:
   - pop 0: v2 seed + smart init (current)
   - pop 1: no v2 seed, random init only
   - pop 2: alternative (e.g. layer-mode-biased smart init)

   Reduce migration_interval so populations stay independent longer.

4. **Realistic gap closure**: +0.4–0.6 pp from diversification + multi-
   strategy populations alone. To close the full 1.59 pp, also need:
   - Block-aware LS operator for BR1#3/#4-type hard instances
   - Possibly layer-decoder bias (G&R 2013 used layer decoders only)

A focused implementation plan is now warranted (next document).

---

## 5. Open questions

- Is the literature's 92.62 % at 30 s budget or longer? Original 2013
  paper experiments often used 60 s or more on 2010-era hardware.
  Need to verify.
- Does the literature use multi-pop with migration? We do (K=3,
  migration every 15 gens). Worth checking the migration policy.
- Is BR1 special — fewer SKUs, more homogeneous boxes — and does it
  reward different polish operators than BR3/5/7?
