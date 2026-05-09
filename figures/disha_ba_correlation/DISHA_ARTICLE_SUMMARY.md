# Disha BA + Correlation — Article Summary

## 1. Plain-English summary

This analysis uses public BIS LBS country-level banking-system exposure data. It builds a
directed exposure network between national banking systems and compares the thresholded
topology with a Barabási-Albert-style preferential-attachment benchmark. It also computes
Pearson correlations between country exposure time series to show which banking systems
moved together more strongly around the Lehman period. The result is useful as a macro
network illustration for an article, but it is not bank-level interbank evidence and it
does not validate repo liquidity contagion.

## 2. Data

- Source: BIS Locational Banking Statistics (LBS), bulk feed `WS_LBS_D_PUB`.
- Aggregation: country-level reporting banking system → counterparty country (sector A).
- Period span: panel covers the full window provided by upstream dataset_dir.
- Residual node `5Q`: already filtered upstream; not present in the country list.

## 3. BA-style comparison

We threshold each period's averaged exposure matrix at the top 15% of
positive weights and compare the thresholded undirected degree sequence to 1000
Barabási-Albert simulated networks of the same N and matched edge density. We report Pearson
correlation between empirical and BA mean-sorted degree sequences, plus a Kolmogorov-Smirnov
distance between the two distributions.

  normal: BA r = 0.941, KS = 0.349, m = 2, Gini = 0.572 — descriptively similar to preferential-attachment-style concentration
  lehman: BA r = 0.945, KS = 0.414, m = 2, Gini = 0.570 — descriptively similar to preferential-attachment-style concentration

This is a **descriptive topology comparison**, not a strict power-law validation: country-level
N (~31) is too small for a clean tail estimate.

## 4. Pearson correlation — what it measures

For each country we sum its outward exposure each quarter (out-strength). Two correlations
are computed:

- **LEVELS**: correlation of raw out-strength time series. Trend-sensitive — a long secular
  growth in claims will inflate level correlations.
- **LOG-CHANGES** (main metric): quarter-over-quarter `log(x + 1)` differences. Removes
  the secular trend; isolates co-movement of shocks.

The headline finding for crisis-window co-movement uses LOG-CHANGES.

**LOW_SAMPLE_WARNING:** the Lehman window contains < 4 usable observations after the log-change transformation. Top change-correlation pairs for the Lehman window are informational only; rely on the sensitivity window 2007Q1–2009Q4 for ranking.

## 5. Normal vs Lehman comparison

- Normal window: 2006Q1 – 2007Q4
- Lehman window: 2008Q3 – 2009Q2
- Sensitivity window (wider): 2007Q1 – 2009Q4

The Lehman window is intentionally illustrative and short; correlation results should be read
as descriptive, not as stable statistical inference.

Top |r| level pairs (Lehman):

  1. CL–TW: r = -1.000
  2. CA–DE: r = +0.998
  3. BR–LU: r = +0.998
  4. IM–TW: r = +0.998
  5. CL–IM: r = -0.997
  6. BR–CH: r = +0.997
  7. CH–LU: r = +0.997
  8. GB–JE: r = +0.995

Top |r| log-change pairs (Lehman):

  1. BE–FI: r = +1.000
  2. IE–TW: r = +1.000
  3. GR–SE: r = +1.000
  4. CH–LU: r = +1.000
  5. CH–IM: r = +1.000
  6. IM–LU: r = +0.999
  7. CL–JE: r = -0.999
  8. CA–FR: r = +0.999

## 6. Risk concentration interpretation

We rank countries by the increase in their mean |Pearson r| of log-changes from the normal
to the Lehman window (then by Lehman |r| and by Lehman total strength as tiebreakers).
This identifies banking systems whose exposure shocks moved more in lock-step with others
around the Lehman period.

Top 10:

  1. CA — Δ|r|_changes = +0.491, |r|_changes_lehman = 0.782
  2. JE — Δ|r|_changes = +0.483, |r|_changes_lehman = 0.773
  3. IM — Δ|r|_changes = +0.481, |r|_changes_lehman = 0.789
  4. DE — Δ|r|_changes = +0.479, |r|_changes_lehman = 0.775
  5. CL — Δ|r|_changes = +0.474, |r|_changes_lehman = 0.777
  6. LU — Δ|r|_changes = +0.469, |r|_changes_lehman = 0.789
  7. TW — Δ|r|_changes = +0.446, |r|_changes_lehman = 0.785
  8. AT — Δ|r|_changes = +0.440, |r|_changes_lehman = 0.732
  9. MO — Δ|r|_changes = +0.413, |r|_changes_lehman = 0.789
  10. FR — Δ|r|_changes = +0.411, |r|_changes_lehman = 0.786

## 7. Limitations

1. Country-level aggregation, not bank-level interbank exposures.
2. Small N (~31 reporting countries).
3. Edge thresholding is sensitive to `--edge-quantile`; results may shift with different cuts.
4. Pearson correlation over short (~4-quarter) windows is statistically fragile.
5. Level correlations are trend-sensitive.
6. Log-change correlations are better for shock co-movement but still descriptive — not causal.
7. Not a bank-level repo contagion model.
8. Not a causal model — these are descriptive co-movement statistics.

## 8. Reproducibility

```
python tools/build_disha_ba_correlation_figures.py \
    --dataset-dir /tmp/bis_real_data/dataset_dir_v2 \
    --output-dir figures/disha_ba_correlation \
    --normal-start 2006Q1 \
    --normal-end 2007Q4 \
    --lehman-start 2008Q3 \
    --lehman-end 2009Q2 \
    --sensitivity-start 2007Q1 \
    --sensitivity-end 2009Q4 \
    --edge-quantile 0.85 \
    --top-n-labels 12 \
    --ba-simulations 1000 \
    --seed 42
```

Git SHA: `e01faaa81f729a0221e5cbc981b3f953b37ae70e`

## 9. Out of scope

This analysis does NOT use Kuramoto / phase-synchronisation modelling. The earlier X-9R
Kuramoto pipeline is unrelated to this artefact and was explicitly excluded per Disha's
request.

This analysis does NOT make bank-level claims. The BIS LBS bulk feed publishes bilateral
cells only at country level; bank-to-bank bilateral data requires supervisory access.

## 10. Suggested 100-word paragraph for Disha

> Using public BIS Locational Banking Statistics, we built a directed network of cross-border
> claims between national banking systems and compared its thresholded topology with a
> Barabási-Albert preferential-attachment benchmark. We then computed Pearson correlations
> between country-level exposure time series for a normal pre-crisis window and the Lehman
> window. Several country pairs show markedly stronger co-movement of exposure changes during
> the Lehman period, and a small set of countries dominate the network's mass. These figures
> illustrate macro banking-network structure descriptively; they are not bank-to-bank
> exposures and do not constitute a validated liquidity-contagion model.
