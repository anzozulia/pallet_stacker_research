# Honest BR Re-measurement — Utilization With Support Enforced

The verification work (`30_verification.md`) surfaced that BR utilization had
been measured with **support enforcement silently disabled** (defect D2). After
fixing D2 (always-on stability) and the block decoder, this re-measures BR1–BR7
**with** support enforced and asks the honest question: once we pack to the same
physical-stability standard the literature uses, do we still "beat 2013 SOTA"?

> **Headline.** **No — the prior "beats G&R/Lim 2013 by 3–7 pp on BR3/5/7" claim
> does not hold.** It compared our **no-support** packing (~92 %) against the
> literature's **full-support** results (~89–93 %) — apples-to-oranges. On the
> like-for-like full-support comparison we are **~1.7 pp behind G&R 2013 on
> average**: badly behind on BR1 (−8.5 pp), slightly behind on BR3 (−1.3 pp),
> and slightly **ahead** on BR5/BR7 (+0.8 / +1.9 pp). The one genuinely good
> result: at full support our packings are **100 % validator-clean** (the
> block-decoder + D2 fixes make enforced-support packing actually valid).

---

## 1. The measurements (n=100, full benchmark)

Protocol: converged BRKGA (patience=20, contention-immune), local search / LNS /
v2-seed off for deterministic, thread-invariant results; `population_size=600`,
`n_populations=3`, `n_modes=6`, `seed=42+instance_id`. Run serially per set so
the numbers are uncontended. **`support_ratio` is the only variable.** Every one
of the 1,200 packings (4 sets × 3 levels × 100) was validator-clean.

| Set | n | sr=0.0 (no support) | sr=0.8 | sr=1.0 (full support) | validator-clean |
|---|---:|---:|---:|---:|:---:|
| BR1 | 100 | 91.19 % | 85.94 % | **84.14 %** | 300/300 |
| BR3 | 100 | 92.77 % | 90.82 % | **89.24 %** | 300/300 |
| BR5 | 100 | 92.86 % | 91.28 % | **89.53 %** | 300/300 |
| BR7 | 100 | 92.71 % | 90.28 % | **87.33 %** | 300/300 |
| **mean** | | **92.38 %** | **89.58 %** | **87.56 %** | **1200/1200** |

**Cost of full support** (sr=0 → sr=1.0): BR1 −7.05, BR3 −3.53, BR5 −3.33,
BR7 −5.38; **mean −4.82 pp**. Stability is not free — and it costs most on the
homogeneous BR1 (few SKUs, uniform sizes), where dense packing relied heavily on
letting boxes overhang/float into otherwise-wasted space.

> Protocol note: these converged-no-LS no-support numbers (sr=0, mean 92.38 %)
> sit slightly below the prior audit's time-bounded-with-LS numbers (BR3 93.84,
> BR5 93.21, BR7 92.88) — a protocol difference, not a regression. The headline
> below is robust to it.

---

## 2. What the literature actually measures (the decisive question)

Whether the prior comparison was fair hinges on whether the cited ~89–93 %
baselines (G&R 2013, Lim 2013, B&R 1995) are reported **with** or **without**
support. A two-agent literature investigation (high confidence, well-sourced)
found:

- **The BR (thpack) data files encode no support constraint.** OR-Library's
  `thpack1–7` specify only container/box dims, an orientation flag, and the
  objective "maximise volume utilisation." Support is a modeling choice each
  author adds on top.
- **The cited baselines are the full-support results.** Bortfeldt & Wäscher's
  *Container Loading Problems — A State-of-the-Art Review* (EJOR 2013) explicitly
  lists **both Bischoff & Ratcliff 1995 and Gonçalves & Resende 2012** in the
  "100 % support" camp. G&R specifically built *"a procedure for joining free
  spaces in the case where full support from below is required."* The +11 pp jump
  from B&R 1995 (~83 %) to G&R (~94 %) is a **like-for-like full-support**
  improvement.
- **The community reports both variants on identical instances.** Zhao's
  Southampton thesis tabulates G&R as two columns — `GR2012S` (stability) and
  `GR2012U` (no stability) — with the no-support (U) numbers consistently
  *higher*. No-support SOTA reaches ~95–96 %.

**Residual ambiguity:** the agents split on whether the *exact* "92.6 % BR1"
figure is the S or U column (resolving it needs the original G&R table cell).
But the verdict is robust either way:

| If the baselines are… | Compare our… | Result |
|---|---|---|
| **full-support** (stronger evidence) | full-support 87.56 % vs ~89.3 % | **behind ~1.7 pp** |
| **no-support** | no-support 92.38 % vs no-support SOTA ~95–96 % | **behind ~3–4 pp** |

On **every** like-for-like axis we trail. The "+3–7 pp over SOTA" gap only ever
existed because it crossed the support / no-support boundary.

---

## 3. Honest like-for-like standing (full-support vs full-support)

Treating the cited G&R/Lim numbers as full-support (the dominant interpretation):

| Set | ours sr=1.0 | G&R 2013 | Δ G&R | Lim 2013 | Δ Lim |
|---|---:|---:|---:|---:|---:|
| BR1 | 84.14 % | 92.6 % | **−8.46** | 93.0 % | −8.86 |
| BR3 | 89.24 % | 90.5 % | **−1.26** | 91.0 % | −1.76 |
| BR5 | 89.53 % | 88.7 % | **+0.83** | 89.3 % | +0.23 |
| BR7 | 87.33 % | 85.4 % | **+1.93** | 86.0 % | +1.33 |
| **mean** | **87.56 %** | **89.3 %** | **−1.74** | **89.8 %** | **−2.27** |

**Reading:**

- **BR1 (homogeneous, 3–7 SKUs): −8.5 pp.** This is the real story of BR1. The
  prior audit reported BR1 only −1.23 pp behind G&R — but that was our no-support
  91.37 % vs their full-support 92.6 %. Honestly, at full support BR1 is **−8.5
  pp** behind. Our extreme-point/DFTRC decoders are weak at dense *fully-
  supported* packing of uniform boxes, where layer-based decoders (G&R's
  strength) excel.
- **BR3 (−1.3 pp): roughly competitive**, marginally behind.
- **BR5 (+0.8) and BR7 (+1.9): genuinely competitive / slightly ahead** — on the
  heterogeneous sets, more SKUs give the multi-decoder search room to find dense
  *and* stable packings. This is the one defensible "we hold our own vs 2013-era
  full-support results" claim, and it survives the honest re-measurement.

---

## 4. Bottom line

- **Do not cite "beats 2013 SOTA."** On the honest, like-for-like full-support
  comparison we are ~1.7 pp behind G&R 2013 on average. The prior claim was an
  artifact of comparing no-support utilization to full-support baselines.
- **What is true and worth stating:** with full support enforced, the algorithm
  is **competitive with 2013-era full-support results on the heterogeneous sets
  (BR5/BR7)**, marginally behind on BR3, and clearly behind on the homogeneous
  BR1.
- **The strongest real result is correctness, not the headline number:**
  1,200 / 1,200 packings validator-clean across all support levels. After the
  block-decoder and D2 fixes the packer produces **physically valid,
  fully-supported** packings — what a real-world tool actually needs, and what
  the inflated no-support number was hiding.
- **For the production service this changes nothing** — the service enforces
  stability and will report honest, valid packings. This section corrects the
  *research narrative* in `29_final_audit.md`, not the product.

> Reproduce: `scripts/_verify/measure_br_support.py --set thpack{1,3,5,7}
> --srs 0.0 0.8 1.0 --n 100`; per-set JSON in `results/checkpoints/`.
