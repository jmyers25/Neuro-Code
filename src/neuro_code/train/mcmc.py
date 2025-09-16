# Change select optimal parameters to using MCMC through pymc

# Steps: CV MCMC
    # Split data into k folds with balanced conditions
    # Fit model and estimate parameters using MCMC for the train data of the fold
        # Store full posterior distributions for each parameter (e.g. alpha_f1)
    # Get median of posterior distributions for each parameter in fold
    # Use these 'best estimates' for the fold to generate predicted choice vector for fold's test data
    # Compare predicted choice vector to true choices and calculate accuracy
        # Store each fold's prediction accuracy (e.g. pred_acc_f1)

# Desired output:
# Per subject -
# Mean accuracy across folds
# Variance of accuracy across folds
# Posteriors for each parameter across folds

# Run for each subject for each model?
# Input: Subject data

import pymc3 as pm
import theano.tensor as tt

def fit_rl_mcmc():
