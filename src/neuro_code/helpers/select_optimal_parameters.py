"""
select_optimal_parameters.py

Multi-start Maximum Likelihood Estimation (MLE) for RL model parameters.

This module provides:
- calculate_neglogprob: negative log-likelihood for a single parameter vector.
- select_optimal_parameters: repeated (multi-start) L-BFGS-B to pick the best MLE.
- A reproducible seeding scheme per fit attempt.

It expects the "new model configs" workflow, i.e., parameter handling and bounds are
centralized in helper utilities:
  - extract_pars(pars_dict)    -> {'fixparams': {...}, 'fitparams': ['alpha', 'beta', ...]}
  - make_pars_dict(spec)       -> canonical dict from string/model name/JSON or dict input
  - get_bounds(pars_dict)      -> bounds array aligned with fitparams order
  - sample_x0(pars_dict)       -> initial x0 array aligned with fitparams order
  - get_model_name(pars_dict)  -> string model label for filenames/logging

Inputs:
  data: DataFrame with columns ['Trial_type','Response','Points_earned'].
  pars: Parameter SPEC (dict or string) understood by your helpers.

Outputs:
  Best-fit parameter dict (only fitted params), plus optional full results table.

Notes:
- The likelihood uses a logistic choice model with overflow-safe computation.
- EV updates follow your asym/sym alpha & exp rules using actual observed rewards.
"""

from __future__ import annotations

import math
import json
import random
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt

# Import helpers from the package (new config workflow)
from neuro_code.helpers.extract_pars import extract_pars, get_model_name, make_pars_dict
from neuro_code.helpers.sample_x0 import sample_x0, get_bounds


# ------------------------------- Core likelihood -------------------------------

def _logistic(val: float) -> float:
    """
    Numerically stable logistic σ(x) = 1 / (1 + exp(-x)), with mild clipping of input.
    """
    # Clip extreme magnitudes to avoid overflow in exp()
    if val > 495.0:
        val = 495.0
    elif val < -495.0:
        val = -495.0
    return 1.0 / (1.0 + math.exp(-val))


def _assemble_all_params(pars_spec: Dict[str, float],
                         fitparams: List[str],
                         x_vec: List[float]) -> Dict[str, float]:
    """
    Merge fixed + fitted params into a single dict for the current evaluation.

    Args:
        pars_spec: Full parameter spec (may include fixed + to-be-fitted keys).
        fitparams: Ordered list of parameter names that are *fitted*.
        x_vec:     Current parameter vector aligned with fitparams.

    Returns:
        Dict mapping every relevant parameter name -> value for this evaluation.
    """
    x0_dict = dict(zip(fitparams, x_vec))

    # Determine symmetric vs asymmetric schema from presence of keys in pars_spec
    # (We rely on extract_pars to give us the right fitparams/fixparams.)
    if 'alpha' in pars_spec:
        all_pars = {'alpha': None, 'beta': None, 'lossave': None}
        all_pars.update({'exp': None} if 'exp' in pars_spec else {'exp_neg': None, 'exp_pos': None})
    else:
        all_pars = {'alpha_neg': None, 'alpha_pos': None, 'beta': None, 'lossave': None}
        all_pars.update({'exp': None} if 'exp' in pars_spec else {'exp_neg': None, 'exp_pos': None})

    # Fill from fixed or fitted
    for p in all_pars:
        all_pars[p] = pars_spec[p] if p in pars_spec and p not in x0_dict else x0_dict.get(p, pars_spec.get(p))

    return all_pars


def calculate_neglogprob(
    x0: List[float],
    df: pd.DataFrame,
    pars_spec: Dict[str, float]
) -> float:
    """
    Compute the negative log-likelihood for one parameter vector x0.

    This implements the (observed) choice likelihood with a logistic choice rule
    and updates expected values (EV) using asymmetric/symmetric learning rates
    and exponents depending on the model spec.

    Args:
        x0:        Parameter vector aligned with 'fitparams' from extract_pars(pars_spec).
        df:        Trial dataframe with columns:
                   - 'Trial_type'   (1..4)
                   - 'Response'     (0=missing/ignore; 1=play; 2=pass)
                   - 'Points_earned' (observed outcome)
        pars_spec: Canonical parameter spec (fixed + potential fit params).

    Returns:
        Negative log-likelihood (float).
    """
    df = df.reset_index(drop=True)
    n = len(df)

    trial_types = df['Trial_type'].to_numpy()
    responses   = df['Response'].to_numpy()
    outcomes    = df['Points_earned'].to_numpy()

    parsed = extract_pars(pars_spec)
    fitparams: List[str] = parsed['fitparams']          # ordered names for x0
    # fixparams: Dict[str, float] = parsed['fixparams'] # not used explicitly here

    # Assemble full param dict for this likelihood evaluation
    P = _assemble_all_params(pars_spec, fitparams, x0)

    alpha     = P.get('alpha')
    alpha_neg = P.get('alpha_neg')
    alpha_pos = P.get('alpha_pos')
    beta      = P['beta']
    lossave   = P['lossave']
    exp_one   = P.get('exp')
    exp_neg   = P.get('exp_neg')
    exp_pos   = P.get('exp_pos')

    EV = np.zeros(4, dtype=float)
    ll = 0.0  # accumulate log-likelihood

    for i in range(n):
        t   = int(trial_types[i]) - 1  # 0..3
        ev  = EV[t]
        rsp = responses[i]
        rwd = outcomes[i]

        # Choice probability (logistic of value*beta; loss-averse scaling for negatives)
        val = (lossave * ev if ev < 0 else ev) * beta
        p_play = _logistic(val)

        # Accumulate log-likelihood for the *observed* response
        if rsp == 0:
            # Missing/neutral: contribute 0 to likelihood (or could mask-out row)
            pass
        elif rsp == 1:
            # Played
            p = p_play
            if p <= 0.0:
                p = 1e-300
            ll += math.log(p)
        elif rsp == 2:
            # Passed
            p = 1.0 - p_play
            if p <= 0.0:
                p = 1e-300
            ll += math.log(p)
        else:
            # Unknown response code: ignore (or raise); here we ignore
            pass

        # Update EV with observed outcome
        if rwd != 0:
            if rwd > ev:
                delta    = rwd - ev
                exponent = exp_one if exp_one is not None else exp_pos
                gain     = alpha   if alpha   is not None else alpha_pos
                pe       = gain * (delta ** exponent)
            elif rwd < ev:
                delta    = ev - rwd
                exponent = exp_one if exp_one is not None else exp_neg
                loss     = alpha   if alpha   is not None else alpha_neg
                pe       = -loss * (delta ** exponent)
            else:
                pe = 0.0
            EV[t] += pe

    # Return negative log-likelihood
    return -float(ll)


# ------------------------------- Multi-start MLE -------------------------------

def select_optimal_parameters(
    data: pd.DataFrame,
    subject: Union[int, str],
    n_fits: int = 50,
    pars: Union[str, Dict[str, float], None] = None,
    save: bool = False,
    output_path: Optional[str] = None,
    method: str = "L-BFGS-B",
    tol: float = 1e-6,
    options: Optional[Dict] = None,
    return_full_table: bool = False,
    base_seed: Optional[int] = None
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:
    """
    Run multi-start MLE and return the best-fitting parameter dictionary.

    Workflow (MLE):
      1) Canonicalize parameter spec (pars) using make_pars_dict().
      2) Use extract_pars() to split into fixed vs. fit parameter names.
      3) For each restart:
           - Sample a starting point x0 via sample_x0().
           - Minimize negative log-likelihood with bounds from get_bounds().
           - Record x0, xopt, objective, seed meta.
      4) Pick the row with the lowest neglogprob and return its xopt as dict.

    Args:
        data:   DataFrame with trials: ['Trial_type','Response','Points_earned'].
        subject: Subject identifier for logging/outputs.
        n_fits:  Number of random restarts (multi-start).
        pars:    Parameter spec (dict or string/JSON understood by make_pars_dict()).
        save:    If True, write a CSV of all fits (sorted by neglogprob).
        output_path: Directory or prefix where CSV will be written (required if save=True).
        method:  scipy.optimize.minimize method (default 'L-BFGS-B').
        tol:     Optimizer tolerance.
        options: Extra options dict passed to minimize (e.g., {'maxiter': 500}).
        return_full_table: If True, also return the full DataFrame of all fits.
        base_seed: If provided, seeds all restarts deterministically; if None,
                   each restart uses an independent random seed.

    Returns:
        - If return_full_table=False:
            Dict[str, float] of best-fit parameters for the *fitted* keys.
          (Fixed params are known from pars and need not be returned.)
        - If return_full_table=True:
            (best_param_dict, results_df)
    """
    if not isinstance(pars, dict):
        pars = make_pars_dict(pars)

    parsed = extract_pars(pars)
    fixparams: Dict[str, float] = parsed['fixparams']
    fitparams: List[str]        = parsed['fitparams']
    model_name: str             = get_model_name(pars)
    bounds                      = get_bounds(pars)  # aligned with fitparams order

    # Pre-allocate results table
    cols = (
        [f'x0_{p}' for p in fitparams] +
        [f'xopt_{p}' for p in fitparams] +
        ['neglogprob', 'sub_id', 'seed']
    )
    results = pd.DataFrame(np.nan, columns=cols, index=range(n_fits))

    # Deterministic seeding across restarts if base_seed provided
    rng = np.random.default_rng(base_seed) if base_seed is not None else np.random.default_rng()

    for i in range(n_fits):
        # Per-restart seed (recorded for reproducibility)
        seed = int(rng.integers(1_000, 2**31 - 1))
        random.seed(seed)
        np.random.seed(seed)

        # Initial guess aligned with fitparams
        x0 = sample_x0(pars)  # must align with 'fitparams'
        x0_dict = dict(zip(fitparams, x0))

        try:
            res = opt.minimize(
                fun=calculate_neglogprob,
                x0=x0,
                args=(data, pars),
                method=method,
                bounds=bounds,
                tol=tol,
                options=(options or {})
            )
            xopt = res.x if res.success else [float('inf')] * len(x0)
        except (OverflowError, FloatingPointError, ValueError):
            # In case of numerical failure, mark as bad
            xopt = [float('inf')] * len(x0)

        xopt_dict = dict(zip(fitparams, xopt))

        # Record results
        for key in fitparams:
            results.at[i, f'x0_{key}']   = x0_dict[key]
            results.at[i, f'xopt_{key}'] = xopt_dict[key]

        # Evaluate final objective safely (avoid re-raising exceptions)
        try:
            nlp = calculate_neglogprob(xopt, data, pars)
        except Exception:
            nlp = float('inf')

        results.at[i, 'neglogprob'] = nlp
        results.at[i, 'sub_id']     = subject
        results.at[i, 'seed']       = seed

    # Sort by objective and optionally save
    results_sorted = results.sort_values(by='neglogprob', ascending=True)
    if save:
        if not output_path:
            raise ValueError("output_path must be provided if save=True.")
        # If output_path is a directory, build a filename; else treat as prefix
        if output_path.endswith(('/', '\\')):
            filepath = f"{output_path}{model_name}_{subject}.csv"
        else:
            filepath = f"{output_path}_{model_name}_{subject}.csv"
        results_sorted.to_csv(filepath, index=False)
        print(f"[MLE] Saved results: {filepath}")

    # Best row -> dict of fitted parameters
    best_row = results_sorted.iloc[0]
    best_fit = {p: float(best_row[f'xopt_{p}']) for p in fitparams}

    if return_full_table:
        return best_fit, results_sorted
    return best_fit


# ------------------------------- Usage example -------------------------------

if __name__ == "__main__":
    # Minimal example of the MLE workflow (adjust paths/pars as needed)
    #
    # 1) Load your trial data (must have Trial_type, Response, Points_earned)
    # 2) Define/choose a parameter spec (string or dict understood by make_pars_dict)
    # 3) Run multi-start L-BFGS-B and get the best parameters

    import argparse

    parser = argparse.ArgumentParser(description="Multi-start MLE for RL model parameters.")
    parser.add_argument("--data", type=str, required=True, help="Path to trial CSV.")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID/label.")
    parser.add_argument("--pars", type=str, required=True,
                        help="Parameter spec: JSON string or a model name understood by make_pars_dict().")
    parser.add_argument("--n-fits", type=int, default=50, help="Number of random restarts.")
    parser.add_argument("--out", type=str, default="", help="Output file/dir prefix (used if --save).")
    parser.add_argument("--save", action="store_true", help="If set, writes a CSV of all fits.")
    parser.add_argument("--base-seed", type=int, default=None, help="Deterministic base seed for restarts.")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    # Allow --pars to be a JSON object or a simple model key handled by make_pars_dict()
    try:
        pars_input = json.loads(args.pars)
    except json.JSONDecodeError:
        pars_input = args.pars  # treat as model key

    best, table = select_optimal_parameters(
        data=df,
        subject=args.subject,
        n_fits=args.n_fits,
        pars=pars_input,
        save=args.save,
        output_path=args.out if args.out else None,
        return_full_table=True,
        base_seed=args.base_seed
    )

    print("[MLE] Best-fit parameters (fitted keys only):")
    for k, v in best.items():
        print(f"  {k}: {v:.6g}")

    if args.save and args.out:
        print("[MLE] Finished and saved results.")