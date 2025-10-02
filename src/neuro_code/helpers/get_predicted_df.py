# --- get_predicted_df.py ---
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional

def get_predicted_df(
    data: pd.DataFrame,
    pars_dict: Dict[str, float],
    rng: Optional[np.random.Generator] = None
) -> pd.DataFrame:
    """
    Simulates model predictions given trial data and parameter values.
    rng: np.random.Generator for reproducible, parallel-safe randomness.
    """
    if rng is None:
        rng = np.random.default_rng()  # independent stream if caller doesn't supply

    data = data.reset_index(drop=True)

    trial_types = data['Trial_type'].to_numpy()
    responses = data['Response'].to_numpy()
    outcomes  = data['Points_earned'].to_numpy()

    EV = np.zeros(4)  # One for each trial type

    # Extract model parameters
    has_symmetric_alpha = 'alpha' in pars_dict
    has_symmetric_exp   = 'exp' in pars_dict

    alpha     = pars_dict.get('alpha', None)
    alpha_pos = pars_dict.get('alpha_pos', None)
    alpha_neg = pars_dict.get('alpha_neg', None)

    exp     = pars_dict.get('exp', None)
    exp_pos = pars_dict.get('exp_pos', None)
    exp_neg = pars_dict.get('exp_neg', None)

    exp_pos_used   = exp if exp is not None else exp_pos
    exp_neg_used   = exp if exp is not None else exp_neg
    alpha_pos_used = alpha if has_symmetric_alpha else alpha_pos
    alpha_neg_used = alpha if has_symmetric_alpha else alpha_neg

    beta     = pars_dict['beta']
    loss_ave = pars_dict['lossave']

    # Reward probabilities per machine
    reward_probs = {
        0: ([100, -10], [0.5, 0.5]),
        1: ([-100, 10], [0.5, 0.5]),
        2: ([495, -5], [0.1, 0.9]),
        3: ([-496, 5], [0.1, 0.9])
    }

    EV_list, PE_list, choice_prob_list, pred_choice_list, pred_reward_list = [], [], [], [], []

    for i in range(len(data)):
        t = int(trial_types[i]) - 1  # Convert to 0-index
        ev = EV[t]

        # Choice prob (logistic/softmax w/ clipping)
        val = (loss_ave * ev * beta) if ev < 0 else (ev * beta)
        if abs(val) > 495:
            val = 495
        # numerically stable logistic: 1 / (1 + exp(-val))
        # but your code used exp(val)/(exp(val)+1); keep equivalent form but safer:
        choice_prob = 1.0 / (1.0 + math.exp(-val))
        choice_prob_list.append(choice_prob)

        # Sample choice
        pred_choice = 1 if rng.random() < choice_prob else 2
        pred_choice_list.append(pred_choice)

        # Sample reward
        if pred_choice == 1:
            rewards, probs = reward_probs[t]
            predicted_reward = rng.choice(rewards, p=probs)
        else:
            predicted_reward = 0.0
        pred_reward_list.append(predicted_reward)

        # PE & EV update
        if predicted_reward != 0:
            if predicted_reward > ev:
                delta = predicted_reward - ev
                pe = alpha_pos_used * (delta ** exp_pos_used)
            elif predicted_reward < ev:
                delta = ev - predicted_reward
                pe = -alpha_neg_used * (delta ** exp_neg_used)
            else:
                pe = 0.0
            EV[t] += pe
            PE_list.append(pe)
        else:
            PE_list.append(np.nan)

        EV_list.append(ev)

    result_df = pd.DataFrame({
        'Trial_type': trial_types,
        'Response': responses,
        'Points_earned': outcomes,
        'EV': EV_list,
        'PE': PE_list,
        'choiceprob': choice_prob_list,
        'pred_choice': pred_choice_list,
        'pred_reward': pred_reward_list
    })
    result_df['pred_choice'] = result_df['pred_choice'].astype(int)
    return result_df


# TODO
#
# [x] Add fixed seed for all RNG in get_predicted_df (rng: np.random.Generator)
# [x] Parallelize parameter sweeps across CPU cores (scripts/run_parallel.py)
# [x] Record reproducibility metadata in outputs (run_id, seed, param_* columns)
# [x] Improve numerical stability of choice prob (logistic: 1/(1+exp(-val)) + val clipping)
#
# Next steps:
# [ ] Wire param grids to model_configs.py
#     - Provide canonical grids per model family (symmetric vs. asymmetric alpha/exp).
#     - Expose presets: "small", "medium", "full" sweeps.
#
# [ ] Add CLI to run_parallel.py
#     - argparse: --data, --out, --n-reps, --max-workers, --base-seed,
#       --write-incremental, --out-dir, --grid-preset (e.g., "small").
#     - Example: python -m scripts.run_parallel --data data/trials.csv --grid-preset small
#
# [ ] HPC integration (Slurm job arrays)
#     - jobs/run_parallel.batch that reads an index to select a param subset.
#     - Optional: split grid into N shards; stitch outputs safely at the end.
#
# [ ] Deterministic trial-level option (optional feature flag)
#     - Make an option to fix RNG per-trial across runs (e.g., rng seeded with base_seed + trial_idx)
#       so different parameter sets see the same stimulus/reward randomness for A/B comparisons.
#
# [ ] Incremental output & resume
#     - Keep write_incremental=True as default for large runs.
#     - Implement a simple "resume" that skips tasks whose CSVs already exist in sim_outputs/.
#
# [ ] Logging & run manifest
#     - Write a manifest.json with: timestamp, git commit, data path, grid, n_reps, seeds, host info.
#     - Optional: per-run logs (duration, rows written) for monitoring/QA.
#
# [ ] Metrics and QA checks
#     - Add small post-run summary (e.g., choiceprob stats, EV/PE ranges, NaN counts).
#     - scripts/summarize_sims.py to aggregate and sanity-check outputs.
#
# [ ] Unit tests (tests/)
#     - Test reproducibility (same seed -> same outputs).
#     - Test shapes/columns for get_predicted_df and run_parallel end-to-end.
#
# [ ] Benchmarking & profiling
#     - Compare single-core vs multi-core throughput on a fixed grid.
#     - Profile hotspots (Python loop vs. numpy ops); consider micro-optimizations if needed.
#
# [ ] Documentation
#     - README: add programmatic example for run_hyperthreaded() usage.
#     - Docstrings: ensure helpers/ modules have clear parameter/return docs.
#
# [ ] Data schema contract
#     - Document required input columns (Trial_type, Response, Points_earned).
#     - Document output columns (EV, PE, choiceprob, pred_choice, pred_reward, run_id, seed, param_*).
#
# [ ] Optional niceties
#     - Progress bar for local runs (tqdm) guarded behind a flag.
#     - Config-driven runs (YAML): parse grids, paths, and settings from a config file.
