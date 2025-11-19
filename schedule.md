# Revision Schedule and Action Plan

## Overview
Goal: revise the manuscript and replication package so that it clearly positions the contribution, rigorously addresses heterogeneity, reports transparent feature engineering, and strengthens validation/robustness in line with the external reviewer’s guidance.

## Phase 1 – Positioning and Contribution (Week 1)
1. **Reframe problem statement and contribution.**
   - Draft a concise motivation paragraph that highlights the gap in the literature (compared to most recent studies) and the novel empirical setting.
   - Add a bullet-point contribution list (theory, data, methodology) early in the paper.
2. **Align hypotheses with positioning.**
   - Ensure each hypothesis explicitly links to measurable outcomes and heterogeneity dimensions that will be estimated later.

## Phase 2 – Heterogeneity Modeling (Weeks 1–3)
1. **Baseline ordered logit/probit.**
   - Re-estimate with ordered logit and ordered probit to provide a transparent baseline.
   - Report coefficients, standard errors, and goodness-of-fit (LL, AIC, BIC, pseudo-R²).
2. **Random-parameter specifications.**
   - Implement random parameters with heterogeneity in means and variances.
   - Explore correlations among random parameters and grouped random parameters where theory motivates shared variation.
   - Add latent-class and scale heterogeneity models for comparison.
3. **Hierarchical / spatial-temporal effects.**
   - Introduce hierarchical random effects (e.g., region → firm → transaction) or spatial-temporal lags when data allow.
   - Test alternative specifications that include spatial kernels or temporal smoothing to capture unobserved clustering.
4. **Diagnostics and inference.**
   - For each model, compute likelihood ratio tests and Wald tests to compare nested specifications.
   - Summarize LL, AIC/BIC, pseudo-R², temporal/spatial stability metrics, and marginal effects in a model comparison table.
   - Report interpretable marginal effects and elasticities for key variables across classes/scales.

## Phase 3 – Validation and Robustness (Weeks 3–4)
1. **Competitive model comparison.**
   - Benchmark against alternative ML/statistical models (e.g., gradient boosting, random forest) with identical features to demonstrate comparative performance.
2. **Out-of-sample validation.**
   - Hold out temporal and spatial folds to test stability; document results.
   - Conduct external validity checks on any available auxiliary datasets.
3. **Threshold sensitivity (TTC/DRAC/PSD).**
   - Perform sensitivity analyses around TTC, DRAC, PSD thresholds; visualize changes in predictions and marginal effects.
4. **Robustness documentation.**
   - Assemble a robustness appendix summarizing all checks, with references to corresponding scripts.

## Phase 4 – Feature Transparency and Documentation (Weeks 2–4)
1. **Feature dictionary & pipeline.**
   - Create a one-page dictionary describing each feature: definition, unit, source, transformation.
   - Provide a pipeline diagram (or step-by-step table) mapping raw data → cleaning → feature construction.
2. **Tables and figures.**
   - Make every table/figure self-contained, reporting N, K, log-likelihood, AIC/BIC, pseudo-R², plus SDs/correlations for random parameters.
   - Include marginal effects/elasticities directly in the relevant tables or in an adjacent panel.
3. **Writing cleanup.**
   - Tighten prose around feature engineering, emphasizing interpretability and theoretical linkage.

## Deliverables (End of Week 4)
1. **Tidy project package.**
   - Data dictionary & access plan (documented constraints, request process).
   - Cleaned codebase with runnable script(s) that reproduce all tables/figures.
   - README updated with run instructions; ensure repository passes lint/tests.
2. **One-page memo.**
   - Summarize datasets, target journal, hypotheses, and planned heterogeneity specifications.
   - Include proposed submission timeline and milestones for co-author feedback.

## Timeline Snapshot
| Week | Focus | Key Outputs |
|------|-------|-------------|
| 1 | Positioning + baseline models | Revised intro/contribution; ordered logit/probit results |
| 2 | Random parameters + feature dictionary draft | Heterogeneity specs implemented; draft dictionary |
| 3 | Hierarchical/spatial-temporal effects + validation setup | Diagnostics tables; cross-validation scripts |
| 4 | Robustness, documentation, memo package | Final tables/figures, tidy package, one-page memo |

## Next Steps
- Circulate this schedule to co-authors for buy-in.
- Assign clear ownership for each phase to ensure accountability.
- Begin Phase 1 immediately to stay on track for the four-week completion window.
