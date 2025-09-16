import copy
import json
import math
import numpy as np
import os
import pandas as pd
import random
from pathlib import Path
from scipy.stats import truncnorm
from argparse import ArgumentParser
from .fit_rl_mcmc import fit_rl_mcmc
from neuro_code.helpers.get_predicted_df from neuro_code.helpers import get_predicted_df
from neuro_code.helpers.extract_pars import get_model_name

# Added checks for pathing
#Set the paths within the instance of a Virtual Enviornment 
#(On Windows Command Prompt): set TODO_PATH = path
# set SERVER_SCRIPTS = path

try:
    todo_path = Path(os.environ['TODO_PATH'])
    server_scripts = Path(os.environ['SERVER_SCRIPTS'])
except KeyError as e:
    raise EnvironmentError(f"Missing environment variable: {e}. Please set TODO_PATH and SERVER_SCRIPTS.")

# --- Argument parsing ---
parser = ArgumentParser()
parser.add_argument("-s", "--subject", required=True, help="subject number")
parser.add_argument("-dp", "--data_path", default=todo_path / 'machine_game/', help="data path")
parser.add_argument("-op", "--output_path", default=server_scripts / 'fit_rl/.cv_fits/', help="output path")
parser.add_argument("-f", "--fold_nums", type=int, default=4, help="number of CV folds")
parser.add_argument("-p", "--pars", required=True, help="parameters dictionary")
args = parser.parse_args()

# --- Initialize arguments ---
subject = args.subject
data_path = Path(args.data_path)
output_path = Path(args.output_path)
fold_nums = args.fold_nums
pars = args.pars

output_path.mkdir(parents=True, exist_ok=True)

# --- Load subject data ---
data = pd.read_csv(data_path / f'ProbLearn{subject}.csv')

model_name = get_model_name(pars)

# --- Remove first five trials per condition ---
data['con_count'] = data.groupby('Trial_type').cumcount()
data = data.query('con_count > 4')

# --- Assign CV fold numbers, balanced per TrialType ---
n_conditions = data['Trial_type'].nunique()
trials_per_condition = data.shape[0] // n_conditions
fold_ids = list(range(1, fold_nums + 1)) * (trials_per_condition // fold_nums)

random_seed = random.randint(1000, 9999999)
random.seed(random_seed)

def assign_fold_nums(df):
    local_fold_ids = fold_ids.copy()
    random.shuffle(local_fold_ids)
    return pd.Series(local_fold_ids[:len(df)], index=df.index)

data['fold_nums'] = data.groupby('Trial_type', group_keys=False).apply(assign_fold_nums)

# --- Run CV folds ---
all_folds = []

for cur_fold in range(1, fold_nums + 1):
    print("***********************************************")
    print(f'Running fold: {cur_fold} for subject: {subject}')
    print("***********************************************")

    if fold_nums > 1:
        train_data = data[data['fold_nums'] != cur_fold].reset_index(drop=True)
        test_data = data[data['fold_nums'] == cur_fold].reset_index(drop=True)
    else:
        train_data = test_data = data.copy()

    # Fit model
    opt_pars_dict = fit_rl_mcmc(data=train_data, subject=subject, pars=pars)

    # Predict
    pred_df = get_predicted_df(data=test_data, pars_dict=opt_pars_dict)

    # Record results
    fold_out = pd.DataFrame([opt_pars_dict]).add_prefix('xopt_')
    fold_out['fold'] = cur_fold
    fold_out['seed'] = random_seed
    fold_out['sub_id'] = subject
    fold_out['pred_acc'] = pred_df['pred_correct'].mean()
    fold_out['model_name'] = model_name

    all_folds.append(fold_out)

# --- Save results ---
all_folds_out = pd.concat(all_folds, ignore_index=True)
all_folds_out.to_csv(output_path / f'CV_{model_name}_{subject}.csv', index=False)

print("***********************************************")
print(f'Saving output for subject: {subject} in {output_path}')
print("***********************************************")
