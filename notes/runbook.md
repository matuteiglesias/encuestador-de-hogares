
## ⚙️ **Runbook: “EPH Income Regression”**

### 1. **Setup**

```bash
cd src/encuestador/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. **Prepare Data**

Run the preprocessing stage (`preprocess.py`):

```python
from encuestador.preprocess import build_training_matrix
build_training_matrix(years=[2023, 2024], paths_cfg={"data_root": "../../data"}, region_df=region_df)
```

This creates files like:
`data/training/EPHARG_train_23.csv`, `EPHARG_train_24.csv`

---

### 3. **Training Workflow**

**A. Baseline Random Forest (archived):**
Defined in `train.py`, stages 1–4; uses `fit_model()`.

**B. HGBR Regressors (current line):**
Run the notebook / script section:

```python
train_data = pd.read_csv('./../../data/training/EPHARG_train_23.csv')
train_data[columnas_pesos] = np.log10(train_data[columnas_pesos].clip(lower=0) + 1)

X_train, X_test, y_train, y_test = prepare_Xy(train_data, x_cols=x_cols4, y_cols=columnas_pesos)
cat_mask = categorical_mask_from_df(X_train)

mo_hgbr = make_multioutput_hgbr_with_mask(cat_mask)
mo_model, mo_metrics = fit_eval_save(mo_hgbr, X_train, y_train, X_test, y_test, out_dir, tag="HGBR_multioutput")
```

---

### 4. **RegressorChain Variant**

Encodes all categoricals and trains dependent regressors:

```python
Xtr_enc, Xte_enc, ytr_enc, yte_enc, pre_enc, _ = prepare_Xy_onehot(
    train_data, x_cols=x_cols4, y_cols=columnas_pesos, categorical_cols=categorical_X
)

chain = make_per_target_chain(order=list(range(len(columnas_pesos))))
chain_model, chain_metrics = fit_eval_save(chain, Xtr_enc, ytr_enc, Xte_enc, yte_enc, out_dir, tag="HGBR_chain")
```

---

### 5. **Evaluation and Diagnostics**

```python
y_pred = mo_model.predict(X_test)
y_true = y_test.values
target_names = y_test.columns

for j, name in enumerate(target_names):
    print(f"{name:10s} | R²={r2_score(y_true[:, j], y_pred[:, j]):.3f} | RMSE={mean_squared_error(y_true[:, j], y_pred[:, j], squared=False):.3f}")
```

Plot:

```python
plt.scatter(y_true[:, j], y_pred[:, j], alpha=0.3)
plt.plot([y_true[:, j].min(), y_true[:, j].max()], [y_true[:, j].min(), y_true[:, j].max()], 'r--')
```

---

### 6. **Persistence**

All models are saved in:

```
../fitted_models_hgbr/
    HGBR_multioutput.joblib
    HGBR_chain.joblib
    HGBR_chain_preprocessor.joblib
```

---

### 7. **Key Safety Checks**

* Always cast categorical columns with `.astype("category")` before training.
* Ensure no NaNs in `X_train` or `y_train` before fitting (`assert_no_nan_df`).
* Use `log1p` or `PowerTransformer` when numeric instability arises in skewed variables.

