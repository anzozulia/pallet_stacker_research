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

## 4. Diagnostic data needed

(Running in background as of this writeup; results to be appended.)

- Saturation curve on BR1#1, #5, #8 at budgets 30/60/120/240 s
- Per-experiment A/B on first 5 BR1 instances (LNS on/off)

When complete, this doc will be amended with empirical numbers and
the experiment plan will be re-ranked.

---

## 5. Open questions

- Is the literature's 92.62 % at 30 s budget or longer? Original 2013
  paper experiments often used 60 s or more on 2010-era hardware.
  Need to verify.
- Does the literature use multi-pop with migration? We do (K=3,
  migration every 15 gens). Worth checking the migration policy.
- Is BR1 special — fewer SKUs, more homogeneous boxes — and does it
  reward different polish operators than BR3/5/7?
