# Review of the LLM "Mosaic Theory" prompt pack (14 November 2026)

**Scope.** This reviews the prompt pack: the single-ticker insider-weighted memo, the cross-sector
screener, and the commodity, healthcare, technology, finance and energy mosaics. It asks what the
pack misses and how to improve it, and compares each fix with what Mosaic Engine already does.

**Method.** Evidence comes from a deep-research workflow:
- 5 search angles, 24 sources fetched, 108 claims extracted.
- 25 claims checked by three independent verifiers each: 24 confirmed, 1 refuted.

Only the confirmed claims are presented as findings below. Topics without confirmed claims are listed
as open questions, not as recommendations.

**Evidence quality.** Most findings rest on one peer-reviewed paper each. Their samples end between
2005 and 2012, returns are gross of costs, and equal-weighted (small-cap) effects are stronger than
value-weighted ones.

## Bottom line

The prompt pack is an **unvalidated scoring rule**:
- weights are hand-set (insiders 35%, weather 40%, FDA 40%);
- thresholds are fixed (+5/+6, RSI<70, short interest<15%);
- there is a discretionary +1 bias toward defense, nuclear and NVDA;
- tiles built from overlapping data are added together.

All of this is applied to signal families whose published edges are smaller, narrower and more decayed
than the weights imply. No step tests whether the score predicts returns after costs.

An LLM filling in the template from web searches adds a further risk: the data and citations it
produces are themselves unverified. This review did not verify claims on that point (see open questions).

## Prioritised gaps (verified evidence)

| # | Gap | Evidence | Fix to the prompts | Mosaic Engine today |
|---|---|---|---|---|
| 1 | Thresholds and weights are never validated, and the number of variants tried isn't counted | Backtest overfitting and selection bias: Bailey & López de Prado, Deflated Sharpe Ratio ([SSRN 2460551](https://papers.ssrn.com/abstract=2460551)); Probability of Backtest Overfitting via CSCV ([SSRN 2326253](https://papers.ssrn.com/abstract=2326253)); multiple-testing hurdle of t>3 for new factors (Harvey, Liu & Zhu, [NBER w20592](https://nber.org/papers/w20592)) | Log every variant (weights, screens, sector templates). Report DSR and PBO. Require t>3 for any new tile. A single hold-out split is not enough | **Present**: DSR/PSR/MinTRL, PBO (now a gate), Hansen SPA and White RC, BH-FDR, pre-registration files, trial-counted strategy set. **Missing**: a t>3 bar specifically for new features (currently BH-FDR at q=0.10) |
| 2 | Arbitrary unequal fixed weights | Equal-weight forecast combinations repeatedly beat estimated "optimal" weights, because estimating the weights adds sampling error ([Oxford Bull. Econ. Stat. 2009](https://econpapers.repec.org/RePEc:bla:obuest:v:71:y:2009:i:3:p:331-355)); none of 14 sample-based portfolio models consistently beat 1/N out of sample (DeMiguel, Garlappi & Uppal, [RFS 2009](https://econpapers.repec.org/RePEc:oup:rfinst:v:22:y:2009:i:5:p:1915-1953)) | Start from equal tile weights, or weights shrunk toward equal. A tilt must beat equal weighting out of sample before it is used | **Weights are estimated** (ridge α=10 ensemble, refit yearly, walk-forward). **Missing**: an explicit equal-weight composite of the signed features as a baseline the ridge model has to beat |
| 3 | Correlated tiles summed together (technicals, flows and options are all built from price and volume) | Redundant signals give their shared component too much weight ([arXiv 2205.04216](https://arxiv.org/pdf/2205.04216)) | Measure correlations between tiles, then cluster, orthogonalise or residualise them before combining | **Partial**: ridge shrinks collinear coefficients; `sue_xmom` / `sue_consensus_xmom` are residualised against momentum. **Missing**: correlation clustering of feature groups before reporting group contributions |
| 4a | All insider trades weighted alike | About 55% of insider trades are routine and earn essentially zero abnormal return. Opportunistic trades only: long-short +82 bp/month value-weighted (t=2.15), +180 bp/month equal-weighted (t=6.07) (Cohen, Malloy & Pomorski, [NBER w16454](https://www.nber.org/papers/w16454.pdf); JF 2012). Sample 1986–2007, gross | Classify each insider as routine or opportunistic from prior-year trading months. Routine trades get weight 0. Keep the insider tile well below 35% until it is validated net of costs | **Present**: `ins_opp_net_90` (+1) and `ins_routine_net_90` (sign 0) in features.py; a pre-registered insider event model (`prereg/insider_event_v1.json`) |
| 4b | Raw "short interest <15%" screen | Days-to-cover largely makes the short ratio redundant. Low-minus-high DTC decile: 1.19%/month (t=6.67) against 0.71% (t=2.57) for the short ratio, 1988–2012, equal-weighted (Hong et al., [NBER w21166](https://www.nber.org/papers/w21166.pdf)). The value-weighted version of this claim was **refuted** 1–2, so large-cap magnitudes should not be quoted | Use a DTC rank as an avoid or short-side filter, not as a bullish signal | **Present**: `si_days_to_cover` (sign −1) from FINRA data |
| 4c | Loose "IV skew / put-call" reading | Stocks with the steepest volatility smirk (OTM put IV minus ATM call IV) underperformed the flattest by about 10.9%/year risk-adjusted, 1996–2005, in-sample and gross (Xing, Zhang & Zhao, [JFQA 2010](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/what-does-the-individual-option-volatility-smirk-tell-us-about-future-equity-returns/ECFD16BA9ACBDC8D577D1BD866FBEA72)). No verified evidence on put-call ratios or unusual volume | Use a defined smirk measure and discount it | **Present but untestable**: `opt_skew_rr` (−1) is snapshot-gated, so it has no point-in-time history and can't be backtested until snapshots accumulate |
| 5 | Published effects treated as current | Across 97 predictors, long-short returns were 26% lower out of sample and 58% lower after publication. Decay was larger for stronger in-sample effects, and what remains sits in illiquid, high-cost names (McLean & Pontiff, [JF 2016](https://Www.Gwern.net/doc/economics/2016-mclean.pdf)) | Cut every literature effect by about half before it informs a weight or threshold. Require net-of-cost evidence in liquid names | **Partial**: per-name cost model (FX + max(½ Corwin-Schultz, 3 bp) + impact) and a self-financing holdings ledger, with no literature prior used. **Missing**: an explicit publication-date split (pre- vs post-publication IC per feature) |

## Concrete changes to the prompt pack

1. **Drop the fixed weights, thresholds and the +1 bias.** Use equal weights within the sector template,
   or estimated, shrunk weights that are backtested and logged. Remove the defense/nuclear/NVDA bias,
   which is a discretionary tilt with no test behind it.
2. **Replace the summed −10…+10 score with a forecast:** expected excess return, with an uncertainty
   band. Calibrate it historically by score decile; Mosaic Engine already checks decile monotonicity
   and conformal coverage. Without that calibration, "+6 = bullish" has no meaning.
3. **Insider section:** keep the Form 4 table, but tag each trade routine or opportunistic. Exclude
   plan trades (10b5-1), grants, exercises and gifts from the net figure, and report opportunistic
   net $ / market cap.
4. **Short interest:** report DTC and the change in DTC, as an avoid filter.
5. **Options:** report a defined 25-delta smirk with its as-of timestamp. Label options data as
   in-sample evidence.
6. **Validation block (new, mandatory):**
   - each tile's historical IC and t-stat, from a tool-verified dataset rather than the LLM's recall;
   - the number of variants tried;
   - the haircut applied;
   - kill criteria.
7. **Data provenance:** every number comes from a tool call against a primary source (EDGAR, FINRA,
   OPRA vendor, EIA/USDA) with a timestamp. The LLM synthesises; it does not supply figures.

## Not covered by verified evidence (open questions)

Sources were fetched for these topics, but none of their claims reached verification, so nothing
here is a finding:
- LLM data and citation hallucination, and out-of-sample evidence on LLM stock picking
  (arXiv 2304.07619, 2505.07078, 2512.23847, 2502.21206, 2508.05201 were fetched).
- Legal: the mosaic-theory doctrine, MNPI boundaries, the first SEC alternative-data enforcement action,
  AI models and non-public information, and the 2023 Rule 10b5-1 amendments (Hunton, BCLP, Akin Gump
  and Skadden notes were fetched).
- 13F lag and alpha decay; RSI and moving-average rules; analyst revisions; cluster buys.
- Commodity weather, CoT and term-structure tiles, and PDUFA/FDA base rates. These carry 40% weights in
  the prompts and still have no verified support.
- Institutional sizing, crowding, capacity and cost practice.

A narrower second verification pass on these topics is the natural next step.

## Mosaic Engine checks against the QR.md checklist

| Item | Status |
|---|---|
| Block-bootstrap uncertainty | Present: stationary-bootstrap IC CIs (`research.py`); net-return, Sharpe and drawdown bootstrap (`baselines.py`) |
| Negative controls / null models | Present: 2,000 matched random portfolios (`baselines.py`); permutation null for the mutual-information screen (`screening.py`). **Missing**: a time-shuffled-signal control run through the full model |
| Known-answer regression test (re-run a past strategy and compare its hash) | **Not found** in `tests/`. Recommended addition |
| t>3 hurdle for new features | Not present (BH-FDR q=0.10 instead) |
| Equal-weight composite baseline | Not present |
