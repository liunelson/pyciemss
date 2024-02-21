# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Test Example

# %%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from typing import NoReturn, Optional, Any
import itertools
from tqdm import tqdm

import os
from pyciemss.PetriNetODE.interfaces import (
    load_and_sample_petri_model,
    load_and_optimize_and_sample_petri_model,
    load_and_calibrate_and_sample_petri_model,
    load_and_calibrate_and_optimize_and_sample_petri_model,
    get_posterior_density_mesh_petri
)

# %%[markdown]
# ## 1. Simulate

# %%
# model_path = "BIOMD0000000955_askenet.json"
model_path = "../../../test/models/AMR_examples/SIDARTHE.amr.json"
# model_path = "../../../test/models/AMR_examples/BIOMD0000000955_askenet.json"
# model_path = "./sir_typed.json"
with open(model_path, 'r') as f:
    model = json.load(f)

# %%
# ## Optimize-Simulate

# %%
num_samples = 5

# timepoints = np.arange(5, dtype = float)
timepoints = np.arange(21, dtype = float) # <--- assertion error when there are timepoints >= intervention time

start_time = timepoints[0] - 1e-5

# Interventions over the parameters
interventions = [(20.0, "beta")]
intervention_bounds = [[0.0], [3.0]]

# Initial condition
start_state = {}
for var in model['semantics']['ode']['initials']:
    start_state[var['target']] = var['expression']
    for param in model['semantics']['ode']['parameters']:
        start_state[var['target']] = start_state[var['target']].replace(param['id'], str(param['value']))
    start_state[var['target']] = float(start_state[var['target']])

# Optimization objective function to be minimized
# e.g. L1 norm of intervention parameters
objective_function = lambda x: np.sum(np.abs(x))

# Quantity of interest that is targeted by the optimization
# tuple of the form (callable function, state variable name, function arguments
# "scenario2dec_nday_average" computes the average of the n last days
qoi = ("scenario2dec_nday_average", "Infected_sol", 2)

optimize_result = load_and_optimize_and_sample_petri_model(
    model_path, 
    num_samples, 
    timepoints = timepoints, 
    interventions = interventions,
    bounds = intervention_bounds,
    qoi = qoi, 
    risk_bound = 10.0,
    objfun = objective_function,
    initial_guess = 1.0, # initial guess of the parameter for the optimizer
    verbose = True,
    start_time = start_time,
    # start_state = start_state
)

# %%
