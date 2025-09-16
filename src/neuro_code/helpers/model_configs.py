# model_configs.py
# Configuration file specifying parameter ranges for each model


# Replace names with underscores, not space, alpha pos, alpha neg as seperate 

import numpy as np

model_configs = {
    "	asym_alpha_sym_exp_loss": {
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp": np.linspace(0.5, 2.0, 4),
            "beta": [0.1, 0.5, 1.0],
            "lossave": [0.5, 1.0]
        }
    },
    "sym_alpha_asym_exp_loss": {
        "params": {
            "alpha": np.linspace(0.1, 1.0, 5),
            "exp_pos": [1.0],
            "exp_neg": [1.5],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [0.5, 1.0]
        }
    },
    "asym_alpha_asym_exp_loss": {
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp_pos": [1.0],
            "exp_neg": [1.5],
            "beta": [1.0],
            "lossave": [0.5]
        }
    },
    "sym_alpha_asym_exp": {  # no lossave (fixed at == 1?)
        "params": {
            "alpha": np.linspace(0.1, 1.0, 5),
            "exp_pos": [1.0],
            "exp_neg": [1.5],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [1] # I think
        }
    },
    "asym_alpha_sym_exp": { # no lossave
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp": np.linspace(0.5, 2.0, 4),
            "beta": [0.1, 0.5, 1.0],
            "lossave": [1] # same as above
        }
    },
    "	asym_alpha_asym_exp": { # no lossave
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp_pos": [1.0],
            "exp_neg": [1.5],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [1]
        }
    },
    "sym_alpha_loss": { # no exp
        "params": {
            "alpha": np.linspace(0.1, 1.0, 5),
            "exp": [1],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [0.5, 1.0]
        }
    },
    "asym_alpha_loss": { # no exp
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp": [1],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [0.5, 1.0]
        }
    },
    "asym_alpha": { # no exp or lossave
        "params": {
            "alpha_pos": [0.2, 0.5],
            "alpha_neg": [0.2, 0.5],
            "exp": [1],
            "beta": [0.1, 0.5, 1.0],
            "lossave": [1]
        }
    },   
}