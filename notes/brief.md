## 🧭 **Briefing: EPH Models Line (Focus Session Summary)**

**Goal:**
Model the EPH training matrices to predict household and individual income variables (`P21`, `P47T`, `PP08D1`, etc.) using categorical and numeric socioeconomic indicators.

**Core idea:**
Incrementally build a multi-stage predictive pipeline:

1. **Stage 1–3:** Classify occupation, inactivity, and income presence.
2. **Stage 4:** Regress monetary variables using a more sophisticated model (HGBR + OneHot + Chains).

**Recent focus (this session):**

* Moved from **RandomForestRegressor** to **HistGradientBoostingRegressor** with:

  * Support for **categorical columns** (`category` dtype or boolean mask).
  * Optional **PowerTransformer / log1p** stabilization for skewed income data.
  * Experimental **RegressorChain** to capture dependencies between income components.
* Established **diagnostic metrics** (`R²`, `RMSE`, `MAE`) and plotting routines.
* Integrated **preprocessing** logic for:

  * `prepare_Xy_onehot()` (numeric + one-hot encoded pipelines).
  * `prepare_Xy()` (categorical-aware DataFrame split).
  * **assert_no_nan_df** and **mask builders** to ensure safe fits.

**Current outcome:**

* Reasonably stable HGBR models with R² between 0.90–0.99 on most income variables.
* Models persist successfully with joblib in `fitted_models_hgbr/`.
* Chain and MultiOutput regressors both operational.
* Diagnostic plotting available via `matplotlib` scatter grids.

**Next steps (2nd pass refinements):**

1. Investigate **feature imbalance** correction (weights, oversampling, reweighting).
2. Explore **monotonic constraints** or **categorical splitting constraints** in HGBR.
3. Add **per-quarter validation** and compare across Q-years.
4. Refactor training into modular CLI or config-based pipeline (so future runs are single-command reproducible).
5. Build evaluation report aggregator (merge all per-target metrics across models).


