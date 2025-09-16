import itertools
import pandas as pd
import numpy as np
import random
import uuid
from typing import Dict, Any
from neuro_code.helpers.model_configs from neuro_code.helpers import model_configs
from neuro_code.helpers.get_predicted_df from neuro_code.helpers import get_predicted_df

# Parralize 
#Parrlize to make a script to run many jobs at a time and then aggregate this data to one place.
#Look into MPI -- Look into hyperthreading for simulate model range ez 
# Figure out how to set up python on the cluster to run the packages
#Steps:
# First run this on the cluster. 
# Extract using external 
# First copy all into the cluster to see if it works there
# Should be a document in rescources to set up python enviornment 
# Watch tutorials on how to use clusters, jobscripts, etc. 
#Today:
# Learn to use cluster and move Code
# Start looking into git and githun to put all my work into my own repo
# Clone her repo and then manipulate all my things into there 

# Filler path to replace with real trial data CSV
TRIAL_DATA_PATH = 'path/to/trial_data.csv'  # TODO: Replace with actual path


def simulate_model(model_name: str) -> pd.DataFrame:
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in model_configs.")

    config = model_configs[model_name]
    param_grid: Dict[str, list] = config['params']
    param_grid = {
    k: v if isinstance(v, (list, np.ndarray)) else [v]
    for k, v in param_grid.items()
}

    # Load shared trial data
    trial_data = pd.read_csv(TRIAL_DATA_PATH)

    # Cartesian product of all parameter combinations
    keys = list(param_grid.keys())
    value_combos = list(itertools.product(*[param_grid[k] for k in keys]))

    all_results = []

    for combo in value_combos:
        pars_dict = dict(zip(keys, combo))
        seed = random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        random.seed(seed)

        df = get_predicted_df(trial_data, pars_dict)
        for key, val in pars_dict.items():
            df[key] = val
        df['model_name'] = model_name
        df['seed'] = seed

        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values(by='Trial_type').reset_index(drop=True)
    return combined_df



if __name__ == '__main__':
    # Example
    model_output = simulate_model('symmetric_model')
    model_output.to_csv('symmetric_model_predictions.csv', index=False)
