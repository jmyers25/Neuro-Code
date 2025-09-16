import math
import numpy as np
import pandas as pd
from typing import Dict



# Seed Tracking + Add to output DF

def get_predicted_df(data: pd.DataFrame, pars_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Simulates model predictions given trial data and parameter values.
    
    Args:
        data (pd.DataFrame): Trial data with 'Trial_type', 'Response', 'Points_earned'.
        pars_dict (dict): Dictionary of model parameters.

    Returns:
        pd.DataFrame: DataFrame with predicted choice, EV, PE, choice prob, and prediction accuracy.
    """
    data = data.reset_index(drop=True)

    trial_types = data['Trial_type'].to_numpy()
    responses = data['Response'].to_numpy()
    outcomes = data['Points_earned'].to_numpy()

    EV = np.zeros(4)  # One for each trial type

    # Extract model parameters
    has_symmetric_alpha = 'alpha' in pars_dict
    has_symmetric_exp = 'exp' in pars_dict

    alpha = pars_dict.get('alpha', None)
    alpha_pos = pars_dict.get('alpha_pos', None)
    alpha_neg = pars_dict.get('alpha_neg', None)

    exp = pars_dict.get('exp', None)
    exp_pos = pars_dict.get('exp_pos', None)
    exp_neg = pars_dict.get('exp_neg', None)

    exp_pos_used = exp if exp is not None else exp_pos
    exp_neg_used = exp if exp is not None else exp_neg
    alpha_pos_used = alpha if has_symmetric_alpha else alpha_pos
    alpha_neg_used = alpha if has_symmetric_alpha else alpha_neg

    beta = pars_dict['beta']
    loss_ave = pars_dict['lossave']

    # Reward probabilities per machine
    reward_probs = {
        0: ([100, -10], [0.5, 0.5]),
        1: ([-100, 10], [0.5, 0.5]),
        2: ([495, -5], [0.1, 0.9]),
        3: ([-496, 5], [0.1, 0.9])
    }

    # Preallocate result lists
    EV_list = []
    PE_list = []
    choice_prob_list = []
    pred_choice_list = []
    pred_reward_list = []

    for i in range(len(data)):
        t = int(trial_types[i]) - 1  # Convert to 0-index
        ev = EV[t]

        # --- Compute choice probability ---
        # Changes Notes:
        # 1. Read in order of stimi presentations - Getting trial type
        # 2. Compute Value
        # 3. Using the value compute choice prob - independant of emprical data -- chocie prob is prob of model beign correct to emprical 
        # 4. Using the prob, compute choice
        # 5. Using that choice -> determine reward (dependant on choice + t-type)
        # 6. Update Ev for Machine
        # Remove resp conditioning. Make val and chocie prob computation universal

        val = loss_ave * ev * beta if ev < 0 else ev * beta
        if abs(val) > 495:
            val = 495
        choice_prob = math.exp(val) / (math.exp(val) + 1)  # softmax func
        choice_prob_list.append(choice_prob)  # Rename to more explicit prob of play

        # Compute Choice based off prob
        pred_choice = 1 if np.random.rand() < choice_prob else 2
        pred_choice_list.append(pred_choice)

        # Compute predicted reward based off choice
        if pred_choice == 1:
            rewards, probs = reward_probs[t]
            predicted_reward = np.random.choice(rewards, p=probs)
        else:
            predicted_reward = 0.0
        pred_reward_list.append(predicted_reward)

        # --- Compute prediction error ---
        # Change to compute off predicted reward (not actual)
        if predicted_reward != 0:
            if predicted_reward > ev:
                delta = predicted_reward - ev
                pe = alpha_pos_used * (delta ** exp_pos_used)
            elif predicted_reward < ev:
                delta = ev - predicted_reward
                pe = -1 * alpha_neg_used * (delta ** exp_neg_used)
            else:
                pe = 0.0
            EV[t] += pe
            PE_list.append(pe)
        else:
            PE_list.append(np.nan)

        # Record state
        EV_list.append(ev)

    # Build output DataFrame
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

    #This was previously the output for response correctness -- must be done out of sim now
    #result_df['response_recode'] = (result_df['Response'] == 1).astype(int)
    #result_df['pred_correct'] = (result_df['pred_choice'] == result_df['response_recode']).astype(int)

    return result_df


# CSV Input rather than pars. Depending on values set, determines what the model is going to be. - Set model name to be i nthe output
# You can tell what model it is based off the params.



#To DO:
# Currently it takes in parameter data and stimuli that are hardcoaded
# want a new thing that simulates the model with args: Model name, depending on model name there will b e a range for each value (beta alhpa etc)
# That should be called by another thing that is hardcoded as a dict of the different models 
# Want the whole range tested 
# the 2nd step will save a csv with all the simulated data 
# WIll need to be doen in parrallels 
# Make tests + benchmark - GEt predicted df is working fine so use that as a test
# Add fixed seed for all RNG in get predicted df
# Essentially we just need to set 2 for loops to run it on all ranges + all models but needs to be run in parallel