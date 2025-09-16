# `neuro_code` package

This directory contains the importable Python package. After installing with `pip install -e .`, you can import modules like:

```python
from neuro_code.helpers.get_predicted_df import get_predicted_df
from neuro_code.train.mcmc import main as run_mcmc
from neuro_code.predict.predict import main as run_predict
```

## Subpackages
- `helpers/`: Shared utilities (simulation, configs, parameter search, etc.)
- `train/`: Model fitting (CV and MCMC)
- `predict/`: Inference/prediction
