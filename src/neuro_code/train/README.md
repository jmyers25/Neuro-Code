# Train

Code for *fitting* RL models.

## Files
- `cv.py`: Cross-validation routines (fold creation, train/eval loops).
- `mcmc.py`: PyMC3/Theano model specification and sampling entry point.
- `make_commands.py`: Generate shell commands for large batches of RL fits.

## Example usage
```bash
python -m neuro_code.train.mcmc --input data/example.csv --out runs/mcmc_out
python -m neuro_code.train.cv --input data/example.csv --kfolds 5 --out runs/cv_out
python -m neuro_code.train.make_commands --input manifest.csv --out scripts/tasks.sh
```
