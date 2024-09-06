# %%[markdown]
# # Test Optimize Interface
#
# Run Optimize and do exploratory analysis of the results.


# %%
import os
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

from pyciemss.ouu.qoi import obs_max_qoi, obs_nday_average_qoi

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
# Models and datasets
MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

# %%
model_url = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")
model_tm = mira.sources.amr.model_from_url(model_url)
model_amr = template_model_to_petrinet_json(model_tm)

# %%
# Run without interventions

start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 10

results = pyciemss.sample(
    model_amr,
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    solver_method = "dopri5"
)

plot_simulate_results(results)

# %%
# Plot



# %%
# Define interventions of different types

int1 = param_value_objective(
    param_name = [],
    start_time = torch.tensor([10.0])
)

int2 = start_time_objective(
    param_name = [],
    param_value = torch.tensor([0.45])
)

int3 = start_time_param_value_objective(
    param_name = []
)

# %%
static_parameter_interventions = intervention_func_combinator(
    [int1, int2],
    [1, 1]
)

static_parameter_interventions = intervention_func_combinator(
    [int1, int2, int3],
    [1, 1, 1]
)

# %%
result = pyciemss.sample(
    model_amr,
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    solver_method = "dopri5",
    static_parameter_interventions = static_parameter_interventions
)

# %%