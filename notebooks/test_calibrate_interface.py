# %%[markdown]
# # Test Dynamic Interventions
#
# Check whether dynamic interventions are triggered multiple times when dealing with non-monotonic trajectories as trigger

# %%
import os
import pyciemss
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
import matplotlib as mpl

import pyciemss.visuals.plots as plots
import pyciemss.visuals.vega as vega
import pyciemss.visuals.trajectories as trajectories

from pyciemss.integration_utils.observation import load_data

from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
    start_time_param_value_objective,
)

# %%
# Models and datasets

MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model3 = os.path.join(MODEL_PATH, "SIR_stockflow.json")

# %%
# Define settings

start_time = 0.0
end_time = 100.0
logging_step_size = 1.0

num_samples = 1

solver_method = "dopri5"

# %%
# Define a dynamic intervention

def make_var_threshold(var: str, threshold: torch.Tensor):
    def var_threshold(time, state):
        return state[var] - threshold
    return var_threshold

dynamic_parameter_interventions = {
    make_var_threshold("I", torch.tensor(150.0)): {"p_cbeta": torch.tensor(0.20)},
    make_var_threshold("R", torch.tensor(800.0)): {"p_cbeta": torch.tensor(0.25)}
}


# %%
result = pyciemss.sample(
    model3, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time, 
    dynamic_parameter_interventions = dynamic_parameter_interventions, 
    solver_method = solver_method
)

# %%
# Plot `sample` result

colors = {i: c for i, (__, c) in enumerate(mpl.colors.TABLEAU_COLORS.items())}

fig, axes = plt.subplots(2, 1, figsize = (10, 10))

for i, ax in enumerate(fig.axes):

    # Parameter trajectories
    if i == 0:

        parameters = [c for c in result["data"].columns if c.split("_")[-1] == "param"]

        for j, p in enumerate(parameters):

            for sample_id in range(num_samples):

                data = result["data"][result["data"]["sample_id"] == sample_id][["timepoint_unknown", p]]

                __ = ax.plot(data["timepoint_unknown"], data[p] / data[p].max(), linewidth = 1.0, alpha = 0.5, color = colors[j])

        __ = plt.setp(ax, ylabel = "Parameters (Normalized)")

        __ = ax.legend([mpl.lines.Line2D([0], [0], linewidth = 1.0, alpha = 0.5, color = colors[j]) for j, __ in enumerate(parameters)], parameters)

    # State trajectories
    if i == 1:

        states = [c for c in result["data"].columns if c.split("_")[-1] == "state"]

        for j, s in enumerate(states):

            for sample_id in range(num_samples):

                data = result["data"][result["data"]["sample_id"] == sample_id][["timepoint_unknown", s]]

                __ = ax.plot(data["timepoint_unknown"], data[s], linewidth = 1.0, alpha = 0.5, color = colors[j])

        __ = plt.setp(ax, ylabel = "State Variables")

        __ = ax.legend([mpl.lines.Line2D([0], [0], linewidth = 1.0, alpha = 0.5, color = colors[j]) for j, __ in enumerate(states)], states)

        # Threshold
        __ = ax.plot([start_time, end_time], [150.0, 150.0], linestyle = "--", color = colors[1])
        __ = ax.plot([start_time, end_time], [180.0, 180.0], linestyle = "--", color = colors[1])


    __ = plt.setp(ax, xlabel = "Timepoints (Days)")

# %%
