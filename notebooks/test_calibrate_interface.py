# %%
# # Test Calibrate
#
# Run Calibrate and do exploratory analysis of the results.
# 

# %%
import os
import pyciemss
import torch
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# from typing import Dict, List, Callable

import pyciemss.visuals.plots as plots
import pyciemss.visuals.vega as vega
import pyciemss.visuals.trajectories as trajectories

from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)

# %%
# Models and datasets
MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model2 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type2_petrinet.json")
model3 = os.path.join(MODEL_PATH, "SIR_stockflow.json")

dataset1 = os.path.join(DATA_PATH, "SIR_data_case_hosp.csv")
dataset2 = os.path.join(DATA_PATH, "traditional.csv")

# %%
start_time = 0.0
end_time = 100.0
logging_step_size = 10.0
num_samples = 100

# %%
data_mapping = {"case": "infected", "hosp": "hospitalized"} # data is mapped to observables
# data_mapping = {"case": "I", "hosp": "H"} # data is mapped to state variables

num_iterations = 10

# Calibrate model1 with dataset1
calibrated_results = pyciemss.calibrate(
    model1, 
    dataset1, 
    data_mapping = data_mapping, 
    num_iterations = num_iterations
)

# %%
calibrated_results

# calibrated_results = {
#     "inferred_parameters": AutoGuideList((0): AutoDelta(), (1): AutoLowRankMultivariateNormal())
#     "loss": <float>
# }

# %%
parameter_estimates = calibrated_results["inferred_parameters"]
parameter_estimates()

# parameter_estimates() = {
#     'persistent_beta_c': tensor(0.4848, grad_fn=<ExpandBackward0>),
#     'persistent_kappa': tensor(0.4330, grad_fn=<ExpandBackward0>),
#     'persistent_gamma': tensor(0.3387, grad_fn=<ExpandBackward0>),
#     'persistent_hosp': tensor(0.0777, grad_fn=<ExpandBackward0>),
#     'persistent_death_hosp': tensor(0.0567, grad_fn=<ExpandBackward0>),
#     'persistent_I0': tensor(9.1598, grad_fn=<ExpandBackward0>)
# }

# %%
# Sample from the parameter estimates and simulate model1 using the sampled configurations
calibrated_sample_results = pyciemss.sample(
    model1, 
    end_time, 
    logging_step_size, 
    num_samples,
    start_time = start_time, 
    inferred_parameters = parameter_estimates
)

calibrated_sample_results["data"].head()

# %%
data0 = calibrated_sample_results["data"][calibrated_sample_results["data"]["timepoint_id"] == 0]

parameters = list(parameter_estimates().keys())
parameter_samples = {p: data0[p + "_param"] for p in parameter_estimates().keys()}

# %%
# Plot sampled configuration
fig, axes = plt.subplots(nrows = 6, ncols = 1, figsize = (8, 12))
fig.tight_layout()

for ax, p in zip(axes, parameters):
    
    __ = ax.hist(parameter_samples[p], density = False, label = p)

    m = np.nanmean(parameter_samples[p])
    s = np.nanstd(parameter_samples[p])
    v = np.var(parameter_samples[p])

    __ = ax.plot([m, m], [0, 2 * np.sqrt(num_samples)], linewidth = 2)
    __ = ax.errorbar(m, np.sqrt(num_samples), yerr = None, xerr = s, capsize = 10.0, elinewidth = 2)

    __ = ax.text(m + 1.5 * s, np.sqrt(num_samples), f"mean = {m:.3f}\nstd = {s:.3f}\nvar = {v:.3f}")

    __ = plt.setp(ax, xlabel = p, ylabel = "Count")

fig.savefig("./figures/test_calibrate_interface.png")

# %%


