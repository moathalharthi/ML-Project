# Setup Guide

Step-by-step instructions for getting `ML-Project.ipynb` running on a fresh machine, plus deployment and troubleshooting notes.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | PyCaret 3.x does not yet support 3.12+ reliably |
| RAM | 4 GB minimum, 8 GB recommended | LightGBM tuning is memory-bound |
| Disk | ~2 GB free | PyCaret pulls many transitive dependencies |
| OS | Linux / macOS / Windows | All work; WSL2 recommended on Windows |

Optional (only if you want to deploy):

- **Docker** 20+ for containerised deployment
- **uvicorn** for the FastAPI endpoint (installed by PyCaret automatically)

---

## 2. Create the environment

###  venv

```bash
python3.11 -m venv pycaret-env
source pycaret-env/bin/activate          # Linux / macOS
pycaret-env\Scripts\activate             # Windows
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip

# Core ML stack
pip install "pycaret[full]"

# Notebook + widget rendering (see Troubleshooting #1)
pip install jupyter ipywidgets

# Models and tuning
pip install lightgbm optuna

# Deployment
pip install gradio fastapi uvicorn
```

> **About `pycaret[full]`**: this pulls scikit-learn, imbalanced-learn, XGBoost, CatBoost, LightGBM, SHAP, plotly, and a dozen others. It's heavy (~1 GB) but you won't have to chase missing imports.

---

## 4. Verify the install

```bash
python -c "import pycaret; print('PyCaret', pycaret.__version__)"
python -c "import lightgbm; print('LightGBM', lightgbm.__version__)"
python -c "import optuna; print('Optuna', optuna.__version__)"
```

All three should print version strings without error.

---

## 5. Run the notebook

```bash
jupyter notebook ML-Project.ipynb
```

Then either *Cell → Run All* or run cells one-by-one with **Shift + Enter**.

The dataset is fetched automatically by `get_data('credit')` on the first run. No manual download needed.

**Expected runtime** on a modern laptop:

- Sections 1–4 (EDA): under 30 seconds
- Section 6 (`compare_models`): 2–4 minutes
- Section 8 (Optuna tuning, 50 trials): 3–6 minutes
- Section 12 (`finalize_model` + deployment artefacts): under 1 minute
- **Total**: roughly 8–12 minutes

---

## 6. Deployment (after running the notebook)

The notebook generates four deployment artefacts in your working directory.

### 6a. Reload the pickled pipeline (Python)

```python
from pycaret.classification import load_model, predict_model
import pandas as pd

model = load_model('credit_model')
new_data = pd.DataFrame([{...}])      # one row of features
predict_model(model, data=new_data)
```

### 6b. FastAPI REST endpoint

```bash
uvicorn credit_api:app --reload
# Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI
```

Sample request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, ...}'
```

### 6c. Docker container

```bash
docker build -t credit-api .
docker run -p 8000:8000 credit-api
```

### 6d. Gradio web UI

The Gradio cell at the bottom of Section 12 launches a browser interface at `http://127.0.0.1:7860`. Users enter the 23 raw features; engineered features are computed automatically.

---

## 7. Troubleshooting

### Issue 1 — `Could not render content for 'application/vnd.jupyter.widget-view+json'`

**Symptom**: A red box appears on cells like `evaluate_model(...)` or `compare_models(...)` with an error mentioning `widget-view+json`.

**Cause**: Your environment is missing the Jupyter widget renderer. Common in:

- VS Code's notebook viewer without the Jupyter extension's widget support enabled
- nbviewer / GitHub previews (these never render widgets)
- JupyterLab without the Jupyter widgets manager installed

**Fix**:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension     # classic Notebook only
```

For JupyterLab:

```bash
pip install jupyterlab ipywidgets
```

The fixed notebook avoids `evaluate_model()` (replaced with explicit `plot_model()` calls) so this won't bite you on Section 11. Earlier sections (e.g. `compare_models()` printing a styled DataFrame) may still need widgets if you want the colourised version.

### Issue 2 — `ModuleNotFoundError: No module named 'pycaret'`

You're either in the wrong virtual environment, or pip installed to a different Python than Jupyter is using.

```bash
# Inside Jupyter, run this in a cell to confirm:
import sys; print(sys.executable)

# Then install into THAT specific Python:
!{sys.executable} -m pip install "pycaret[full]"
```

### Issue 3 — `compare_models` is extremely slow

Symptom: takes 15+ minutes on a small machine.

**Fix**: in Section 6, restrict the model list:

```python
top_models = compare_models(
    sort='AUC',
    include=['lr', 'dt', 'rf', 'lightgbm', 'xgboost']   # only the top 5 candidates
)
```

This skips the slowest models (GP, MLP, SVM) without losing the eventual winner.

### Issue 4 — `optuna` not installed → tuning falls back to random search

Not actually a bug — the notebook's `try/except` falls back gracefully — but if you want Bayesian search:

```bash
pip install optuna
```

### Issue 5 — Gradio launches but the browser shows "connection refused"

If you're running on a remote server, expose the port:

```python
gr.Interface(...).launch(server_name="0.0.0.0", server_port=7860, share=False)
```

If you want a public URL (handy for quick demos, **never** for real PII):

```python
gr.Interface(...).launch(share=True)
```

### Issue 6 — `RuntimeWarning: invalid value encountered in divide` during feature engineering

A row has both `avg_bill = 0` and `avg_payment = 0`. Already handled by the `.clip(lower=1)` and `.clip(upper=5)` calls in Section 3 — the warning is cosmetic. Suppress it explicitly if it bothers you:

```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

---

## 8. Reproducibility

The notebook fixes `session_id=42` in `setup()` so train/test splits, SMOTE samples, and CV folds are deterministic. Hyperparameter tuning is also seeded.

That said, **LightGBM's exact final coefficients can differ very slightly across machines** due to floating-point ordering in parallel histogram building. Headline metrics (AUC, Recall, F1, Accuracy) should match within ±0.005 across runs on the same OS/CPU.

