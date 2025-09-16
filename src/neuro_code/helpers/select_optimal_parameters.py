import math
import numpy as np
import pandas as pd
import random
import scipy.optimize
from scipy.stats import truncnorm
from typing import Dict, List, Union
from .extract_pars from neuro_code.helpers import extract_pars, get_model_name, make_pars_dict
from .sample_x0 from neuro_code.helpers import sample_x0, get_bounds


def calculate_neglogprob(x0: List[float], df: pd.DataFrame, pars: Dict[str, float]) -> float:
    df = df.reset_index(drop=True)

    trial_types = df['Trial_type']
    responses = df['Response']
    outcomes = df['Points_earned']
    EV = [0.0] * 4
    choiceprob = np.zeros(len(df))

    parsed = extract_pars(pars)
    fixparams = parsed['fixparams']
    fitparams = parsed['fitparams']
    x0_dict = dict(zip(fitparams, x0))

    if 'alpha' in pars:
        all_pars = {'alpha': None, 'beta': None, 'lossave': None}
        all_pars.update({'exp': None} if 'exp' in pars else {'exp_neg': None, 'exp_pos': None})
    else:
        all_pars = {'alpha_neg': None, 'alpha_pos': None, 'beta': None, 'lossave': None}
        all_pars.update({'exp': None} if 'exp' in pars else {'exp_neg': None, 'exp_pos': None})

    for par in all_pars:
        all_pars[par] = pars[par] if par in fixparams else x0_dict[par]

    alpha = all_pars.get('alpha')
    alpha_neg = all_pars.get('alpha_neg')
    alpha_pos = all_pars.get('alpha_pos')
    beta = all_pars['beta']
    lossave = all_pars['lossave']
    exp = all_pars.get('exp')
    exp_neg = all_pars.get('exp_neg')
    exp_pos = all_pars.get('exp_pos')

    for i in range(len(df)):
        t = int(trial_types[i]) - 1
        ev = EV[t]
        resp = responses[i]
        reward = outcomes[i]

        # Compute choice probability
        val = (lossave * ev if ev < 0 else ev) * beta
        if resp == 0:
            choiceprob[i] = 1
        elif resp == 1:
            choiceprob[i] = math.exp(val) / (math.exp(val) + 1)
        elif resp == 2:
            choiceprob[i] = 1 - math.exp(val) / (math.exp(val) + 1)

        # Compute prediction error and update EV
        if reward != 0:
            if reward > ev:
                delta = reward - ev
                exponent = exp if exp is not None else exp_pos
                gain = alpha if alpha is not None else alpha_pos
                pe = gain * delta ** exponent
            elif reward < ev:
                delta = ev - reward
                exponent = exp if exp is not None else exp_neg
                loss = alpha if alpha is not None else alpha_neg
                pe = -loss * delta ** exponent
            else:
                pe = 0.0
            EV[t] += pe

    # Compute negative log-likelihood
    choiceprob = np.clip(choiceprob, 1e-8, 1 - 1e-8)
    return -np.sum(np.log(choiceprob))


def select_optimal_parameters(
    data: pd.DataFrame,
    subject: Union[int, str],
    n_fits: int = 50,
    pars: Union[str, Dict[str, float]] = None,
    save: bool = False,
    output_path: str = None
) -> Dict[str, float]:
    """
    Runs optimization multiple times to select best-fitting model parameters.

    Args:
        data: Input data frame with subject behavior.
        subject: Subject identifier.
        n_fits: Number of random restarts.
        pars: Parameter dictionary or JSON string.
        save: Whether to save the output to CSV.
        output_path: File path to save the results (required if save=True).

    Returns:
        Dictionary of optimal parameters.
    """
    if not isinstance(pars, dict):
        pars = make_pars_dict(pars)

    parsed = extract_pars(pars)
    fixparams = parsed['fixparams']
    fitparams = parsed['fitparams']
    model_name = get_model_name(pars)
    bounds = get_bounds(pars) # Potentially make this to fitparam since bounds only cares about fitparam -- potential issue in .optimize.minimize for accepted inputs

    cols = (
        [f'x0_{p}' for p in sorted(pars)] +
        [f'xopt_{p}' for p in sorted(pars)] +
        ['neglogprob', 'sub_id', 'seed']
    )
    results = pd.DataFrame(np.nan, columns=cols, index=range(n_fits))

    for i in range(n_fits):
        seed = random.randint(1000, 99999999)
        random.seed(seed)
        np.random.seed(seed)

        x0 = sample_x0(pars)
        x0_dict = dict(zip(fitparams, x0))

        print("Sampled starting parameters:", x0_dict)

        try:
            res = scipy.optimize.minimize(
                calculate_neglogprob,
                x0,
                args=(data, pars),
                method="L-BFGS-B",
                bounds=bounds,
                tol=1e-6
            )
            xopt = res.x if res.success else [float('inf')] * len(x0)
        except OverflowError:
            xopt = [float('inf')] * len(x0)

        xopt_dict = dict(zip(fitparams, xopt))
        print("Estimated optimal parameters:", xopt_dict)

        # Record results
        for key in fitparams:
            results.at[i, f'x0_{key}'] = x0_dict[key]
            results.at[i, f'xopt_{key}'] = xopt_dict[key]
        for key in fixparams:
            results.at[i, f'x0_{key}'] = pars[key]
            results.at[i, f'xopt_{key}'] = pars[key]

        results.at[i, 'neglogprob'] = calculate_neglogprob(xopt, data, pars)
        results.at[i, 'sub_id'] = subject
        results.at[i, 'seed'] = seed

    if save:
        if not output_path:
            raise ValueError("Output path must be provided if save=True.")
        filepath = f"{output_path}{model_name}_{subject}.csv"
        results.sort_values(by='neglogprob').to_csv(filepath, index=False)
        print(f"Estimated parameters saved to: {filepath}")

    best_row = results.sort_values(by='neglogprob').filter(regex='xopt_').iloc[0]
    opt_pars_dict = {col.replace('xopt_', ''): best_row[col] for col in best_row.index}

    return opt_pars_dict





pars2 = {
    'alpha_pos':  0.8,
    'alpha_neg': 0.1,
    'beta':  5.0,
    'lossave': 0.6,
    'exp': 1.0,
    'Sub_id': 'hi/lo/hi/hi',
    'model': 'alpha_pos_alpha_neg_beta_lambda'
}
pars3 = {
    'alpha_pos':  0.8,
    'alpha_neg': 0.1,
    'beta':  1.0,
    'lossave': 0.5,
    'exp': 1.0,
    'Sub_id': 'hi/lo/lo/hi',
    'model': 'alpha_pos_alpha_neg_beta_lambda'
}