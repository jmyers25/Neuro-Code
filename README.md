# Neuro-Code

**Neuro-Code** is a minimal, refactored package for fitting and evaluating reinforcement‑learning (RL) models used in a neuroscience context. 
It bundles training (cross‑validation and MCMC fitting), prediction/inference, and reusable helper utilities into a clean, importable Python package.

---

## Quick Start

### Option A — pip (editable install)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate neuro-code
```

### Verify installation
```bash
python -c "import neuro_code, sys; print('neuro_code OK')"
```

---

## How the repository is organized

```
Neuro-Code/
├─ src/neuro_code/                # The importable Python package
│  ├─ helpers/                    # Shared utilities for simulation, config, sampling, etc.
│  ├─ train/                      # Model fitting code (CV, MCMC) and command builders
│  └─ predict/                    # Prediction / inference scripts
├─ scripts/                       # Shell helpers and task lists (human-run CLIs)
├─ jobs/                          # Job scheduler files (e.g., Slurm .batch)
├─ data/                          # Small sample data / data notes (no large files)
├─ notebooks/                     # (Optional) Jupyter notebooks for demos/EDA
├─ tests/                         # (Optional) Unit tests
├─ requirements.txt               # Minimal pip dependencies
├─ environment.yml                # Conda environment spec
├─ pyproject.toml                 # Packaging metadata for `pip install -e .`
└─ .gitignore                     # Keeps the repo clean
```

### Folder purposes

- **`src/neuro_code/`**: The core importable package. Keeps imports stable (`from neuro_code.helpers...`) and is ready for packaging.
- **`src/neuro_code/helpers/`**: Reusable utilities for model configuration, simulation, parameter selection, etc.
- **`src/neuro_code/train/`**: Everything related to *fitting* models—cross‑validation procedures, MCMC fitting, and helper scripts to generate batch command lines.
- **`src/neuro_code/predict/`**: Scripts to *apply* trained models to new data and generate predictions.
- **`scripts/`**: Human‑run shell helpers and task lists that orchestrate training/prediction runs.
- **`jobs/`**: Cluster job descriptors (e.g., Slurm `.batch`) to schedule work on HPC resources.
- **`data/`**: Placeholders and documentation for small example inputs. Do not put large/private data under version control.
- **`notebooks/`**: Optional notebooks for EDA, sanity checks, figures for reports, etc.
- **`tests/`**: Optional unit tests if you want CI/regression checks.

---

## File-by-file map and responsibilities

### Package: `src/neuro_code/helpers/`

- **`extract_pars.py`**  
  Utilities to extract/transform parameter dictionaries or arrays from fits or configs into canonical formats.

- **`get_predicted_df.py`**  
  Core computation of prediction‑time quantities (e.g., action values, choice probabilities, EV/PE). Includes overflow‑safe probability calculations.

- **`model_configs.py`**  
  Central store for model and hyperparameter configurations. Define model names, parameter bounds, priors/defaults, etc.

- **`sample_x0.py`**  
  Functions to sample initial parameter vectors (e.g., for multi‑start optimization or MCMC initialization).

- **`select_optimal_parameters.py`**  
  Routines to repeatedly sample candidate initializations and minimize the negative log‑likelihood (e.g., L‑BFGS‑B), selecting the best solution.

- **`simulate_all_models.py`**  
  Batch utilities to simulate multiple model families or settings across grids/ranges for comparison or synthetic data checks.

- **`simulate_model_range.py`**  
  Simulation over specified parameter ranges for a single model; produces outputs suitable for sweep/ablation analyses.

- **`test_predictions.py`**  
  Small diagnostics/sanity tests for prediction pipelines; helpful to verify expected ranges and shapes before large jobs.

### Package: `src/neuro_code/train/`

- **`cv.py`**  
  Cross‑validation utilities to split data, train on folds, and evaluate generalization. Intended for model selection and hyperparameter tuning.

- **`mcmc.py`**  
  MCMC-based fitting (via PyMC3/Theano); constructs probabilistic model, defines priors/likelihood, runs samplers, and stores posterior summaries.

- **`make_commands.py`**  
  Helper to generate shell commands for large batches of RL fits (e.g., enumerate datasets, subjects, models, and write commands to a task list).

### Package: `src/neuro_code/predict/`

- **`predict.py`**  
  Applies trained parameters/models to new data to generate predicted choices, probabilities, and metrics. Can be used as a module or run as a script.

### Shell helpers: `scripts/`

- **`run_fit_rl.sh`**  
  A convenience script to invoke training routines (e.g., call `python -m neuro_code.train.mcmc` with appropriate arguments).

- **`pred_rl_task_list.sh`**  
  A task list or orchestrator for batch prediction jobs; often used together with cluster schedulers or GNU parallel.

### Cluster jobs: `jobs/`

- **`fit_rl.batch`**  
  Slurm (or other scheduler) batch script for training runs. Adjust resources, modules, and environment before submission.

- **`run_pred_rl.batch`**  
  Slurm batch script for running predictions at scale.

### Data notes: `data/`

- **`README.md`**  
  Explains expected data layout, file naming, and where to obtain real datasets. Keep only tiny example inputs here if necessary.

---

## Typical workflows

### 1) Fit a model (local or cluster)
```bash
# Local example with defaults
python -m neuro_code.train.mcmc --input path/to/data.csv --out runs/mcmc_out

# Cross-validation
python -m neuro_code.train.cv --input path/to/data.csv --kfolds 5 --out runs/cv_out
```

### 2) Generate predictions
```bash
python -m neuro_code.predict.predict --input path/to/new_data.csv     --params runs/mcmc_out/best_params.json --out runs/predictions.csv
```

### 3) Batch jobs (cluster)
```bash
# Edit jobs/fit_rl.batch to set your conda env/module loads
sbatch jobs/fit_rl.batch
```

---

## Dependencies

- Core:
  - `numpy`, `pandas`, `scipy`
- Probabilistic modeling:
  - `pymc3`, `theano`  *(or `theano-pymc` under conda)*

See `requirements.txt` or `environment.yml` for installable specs.

---

## Contributing & style

- Place reusable code in `helpers/`.
- Keep train/predict modules CLI‑friendly (use `argparse` / `main()`).
- Prefer small, documented functions. Add tests under `tests/` when possible.

---

## License & citation

Add your license details here if distributing publicly. If you use this code in publications, please cite appropriately.
