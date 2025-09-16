# Helpers

Shared utilities used by training and prediction pipelines.

## Files
- `extract_pars.py`: Convert parameter artifacts into standardized dicts/arrays.
- `get_predicted_df.py`: Compute action values, choice probabilities, EV/PE; returns a tidy DataFrame for downstream analysis.
- `model_configs.py`: Central registry of model definitions and hyperparameters.
- `sample_x0.py`: Sample initial parameter vectors for optimization or MCMC.
- `select_optimal_parameters.py`: Multi-start optimization wrapper selecting the best solution (e.g., by NLL).
- `simulate_all_models.py`: Run simulation suites across multiple model families/settings.
- `simulate_model_range.py`: Parameter sweep for a single model (for ablations/what-ifs).
- `test_predictions.py`: Lightweight checks to validate prediction outputs.
