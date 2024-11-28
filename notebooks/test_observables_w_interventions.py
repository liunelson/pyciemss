# %%markdown
# # Test Observations with Intervention Policies
# 
# Previously, observables are not updated when there are interventions.

# %%
import os
import json
import pyciemss
import torch
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl

import mira
from mira.modeling.amr.petrinet import template_model_to_petrinet_json
from mira.sources.amr.petrinet import template_model_from_amr_json

from pyciemss.integration_utils.observation import load_data

from pyciemss.integration_utils.intervention_builder import (
    intervention_func_combinator,
    param_value_objective,
    start_time_objective,
    start_time_param_value_objective,
)

# %%
# Plot pyciemss.simulate results
def plot_simulate_results(results: dict) -> None:

    colors = mpl.colormaps["tab10"](range(10))
    df = results["data"].groupby(["timepoint_id"]).mean()

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))

    i = 0
    names = []
    for c in results["data"].columns:
        if c.split("_")[-1] == "state":
            for n in range(num_samples):
                df = results["data"][results["data"]["sample_id"] == n]
                __ = ax.plot(df["timepoint_unknown"], df[c], label = c, alpha = 0.5, color = colors[i, :])
            
            names.append(c)
            i += 1

    __ = ax.legend(handles = [mpl.lines.Line2D([0], [0], alpha = 0.5, color = colors[j, :], label = names[j]) for j in range(i)])

    return None

# %%
# Load model
with open('./data/SIDARTHE model with observables (lamb).json', 'r') as fp:
    model_amr = json.load(fp)
    model_mmt = template_model_from_amr_json(model_amr)

# %%
# Define assumptions

start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 10

static_parameter_interventions = {
    torch.tensor(4.0): {
        "alpha": torch.tensor(0.422),
        "beta": torch.tensor(0.0057),
        "delta": torch.tensor(0.0057),
        "gamma": torch.tensor(0.285)
    },
    torch.tensor(12.0): {
        "epsilon": torch.tensor(0.143)
    },
    torch.tensor(22.0): {
        "alpha": torch.tensor(0.360),
        "beta": torch.tensor(0.005),
        "delta": torch.tensor(0.005),
        "gamma": torch.tensor(0.200),
        "zeta": torch.tensor(0.034),
        "eta": torch.tensor(0.034),
        
        "mu": torch.tensor(0.008),
        "nu": torch.tensor(0.015),
        "lambda": torch.tensor(0.08),
        "rho": torch.tensor(0.017),
        "kappa": torch.tensor(0.017),
        "xi": torch.tensor(0.017),
        "sigma": torch.tensor(0.017),
    },
    torch.tensor(28.0): {
        "alpha": torch.tensor(0.210),
        "gamma": torch.tensor(0.110)
    },
    torch.tensor(38.0): {
        "epsilon": torch.tensor(0.200),
        "rho": torch.tensor(0.020),
        "kappa": torch.tensor(0.020),
        "xi": torch.tensor(0.020),
        "sigma": torch.tensor(0.010),
        "zeta": torch.tensor(0.025),
        "eta": torch.tensor(0.025)
    }
}

# %%
# Run

result = pyciemss.sample(
    model_amr,
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    solver_method = "dopri5",
    # static_parameter_interventions = static_parameter_interventions
)

plot_simulate_results(result)

# %%
