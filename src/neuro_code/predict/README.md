# Predict

Apply trained parameters/models to new data.

## Files
- `predict.py`: CLI/module for generating predicted choices/probabilities.

## Example usage
```bash
python -m neuro_code.predict.predict --input data/new_data.csv --params runs/mcmc_out/best_params.json --out runs/preds.csv
```
