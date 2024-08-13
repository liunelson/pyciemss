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

import pyciemss.visuals.plots as plots
import pyciemss.visuals.vega as vega
import pyciemss.visuals.trajectories as trajectories

from pyciemss.integration_utils.observation import load_data

from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)

from pyciemss.ouu.qoi import obs_nday_average_qoi

# %%
# Models and datasets
MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model1 = os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json")

dataset1 = os.path.join(DATA_PATH, "SIR_data_case_hosp.csv")
d = load_data(dataset1)
d = {"timepoint": d[0]} | d[1]
d = {k: v.numpy() for k, v in d.items()}
dataset1_df = pd.DataFrame(d)

# %%
# Common Settings

start_time = 0.0
end_time = 60.0
logging_step_size = 1.0
num_samples = 100

# %%
# Define interventions
static_parameter_interventions = {
    10.0: {"beta_c": torch.tensor(0.2)},
    15.0: {"gamma": torch.tensor(0.4)}
}

# %%
# Sample before optimize

result_preoptimize = pyciemss.sample(
    model1,
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    static_parameter_interventions = static_parameter_interventions,
    solver_method="dopri5"
)

# %%
# Optimize settings

num_samples_ouu = 100
maxiter = 5
maxfeval = 5

# %%
# Define interventions

observed_params = ["I_state"]
intervention_time = [torch.tensor(10.0), torch.tensor(15.0)]
intervened_params = ["beta_c", "gamma"]
param_current = [0.35, 0.2]
initial_guess_interventions = [0.2, 0.4]
bounds_interventions = [[0.1, 0.1], [0.5, 0.5]]

static_parameter_interventions = param_value_objective(
    param_name = intervened_params,
    start_time = intervention_time,
)

risk_bound = 5000
qoi = lambda y: obs_nday_average_qoi(y, observed_params, 1)
objfun = lambda x: np.sum(np.abs(x - param_current))

# %%
result_optimize = pyciemss.optimize(
    model1,
    end_time,
    logging_step_size,
    qoi,
    risk_bound,
    static_parameter_interventions,
    objfun,
    initial_guess_interventions = initial_guess_interventions,
    bounds_interventions = bounds_interventions,
    start_time = start_time,
    n_samples_ouu = num_samples_ouu,
    maxiter = maxiter,
    maxfeval = maxfeval,
    solver_method = "dopri5",
)

print(result_optimize)

# %%
print("Optimal intervention: ", static_parameter_interventions(result_optimize["policy"]))

# %%
# Run Sample after Optimize

result_postoptimize = pyciemss.sample(
    model1,
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    static_parameter_interventions = static_parameter_interventions(result_optimize["policy"]),
    solver_method="dopri5"
)

# %%
fig, axes = plt.subplots(3, 1, figsize = (12, 12))

data = result_preoptimize["data"].groupby(["timepoint_id"]).mean()

x = data["timepoint_unknown"]
y = data["persistent_beta_c_param"]
__ = axes[0].plot(x, y, linestyle = "--", color = "b", label = "beta_c (pre-optimized)")

y = data["persistent_gamma_param"]
__ = axes[0].plot(x, y, linestyle = "--", color = "r", label = "gamma (pre-optimized)")

x = data["timepoint_unknown"]
y = data["I_state"]
__ = axes[1].plot(x, y, linestyle = "--", color = "g", label = "I (pre-optimized)")

data = result_postoptimize["data"].groupby(["timepoint_id"]).mean()

x = data["timepoint_unknown"]
y = data["persistent_beta_c_param"]
__ = axes[0].plot(x, y, linestyle = "-", color = "b", label = "beta_c (pre-optimized)")

y = data["persistent_gamma_param"]
__ = axes[0].plot(x, y, linestyle = "-", color = "r", label = "gamma (pre-optimized)")

x = data["timepoint_unknown"]
y = data["I_state"]
__ = axes[1].plot(x, y, linestyle = "-", color = "g", label = "I (pre-optimized)")

# %%




# %%
# Plot parameter distributions

parameters = list(parameter_estimates().keys())
num_parameters = len(parameters)

fig, axes = plt.subplots(nrows = num_parameters, ncols = 1, figsize = (8, 15))
fig.suptitle("Parameter Distributions")
fig.tight_layout()

for ax, p in zip(axes, parameters):
    

    # Pre-/post-calibrate trajectories
    for i, (result_, label, color) in enumerate(zip((result_precalibrate, result_postcalibrate), ("Prior", "Posterior"), (cmap[0, :], cmap[1, :]))):

        # Filter the result dataset
        result = result_["data"][result_["data"]["timepoint_id"] == 0]
        samples = result[p + "_param"]
        
        # Compute histogram (use the pre-calibrate bins for the post-calibrate dataset)
        if i == 0:
            h, b = np.histogram(samples, density = False)
            w = b[1] - b[0]
        else:
            h, b = np.histogram(samples, density = False, bins = b)


        # x coor to center the bars
        x = 0.5 * (b[1:] + b[0:-1])

        if i == 0:
            __ = ax.bar(x, h + 0.1, width = 0.9 * w, align = "center", label = label, alpha = 0.7)
        else:
            __ = ax.bar(x, h + 0.1, width = 0.5 * w, align = "center", label = label, alpha = 0.7)


    # Legend
    l = [
        mpl.patches.Patch(facecolor = cmap[0, :], alpha = 0.7, label = "Prior"),
        mpl.patches.Patch(facecolor = cmap[1, :], alpha = 0.7, label = "Posterior"),
    ]
    __ = ax.legend(handles = l)

    # Axis labels
    xlabel = "_".join(p.split("_")[1:])
    __ = plt.setp(ax, xlabel = xlabel, ylabel = "Count")


fig.savefig("./figures/test_calibrate_interface_parameter_distributions.png")

# %%
# Plot errors

# RNG seed
rng = np.random.default_rng(0)


cmap = mpl.colormaps["tab10"](range(10))
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 12))
fig.suptitle("State Variable Errors")
# fig.tight_layout()

for ax, (train_data_state, model_state) in zip(axes, data_mapping.items()):

    # Interpolate training dataset over same timepoints as Calibrate results
    x_train = dataset1_df["timepoint"]
    y_train = dataset1_df[train_data_state]
    x_postcalibrate = result_postcalibrate["data"][result_postcalibrate["data"]["sample_id"] == 0]["timepoint_unknown"]
    x_postcalibrate = x_postcalibrate[(x_postcalibrate >= min(x_train)) & (x_postcalibrate <= max(x_train))]
    y_train_interp = np.interp(x_postcalibrate, x_train, y_train, left = None, right = None)

    # Repeat interpolated y values to align with Calibrate results
    y_train_interp_repeat = pd.concat([pd.DataFrame(y_train_interp)] * num_samples, axis = 0)
    y_train_interp_repeat = y_train_interp_repeat.reset_index(drop = True)[0]

    # Get Calibrate result column name of the model state variable (as named in the data mapping)
    if "_".join([model_state, "state"]) in result.columns:
        l = "_".join([model_state, "state"])
    elif "_".join([model_state, "observable", "state"]) in result.columns:
        l = "_".join([model_state, "observable", "state"])
    else:
        l = None

    if ~isinstance(l, type(None)):

        # Compute error (mean absolute error)
        result = result_postcalibrate["data"][result_postcalibrate["data"]["timepoint_unknown"].isin(x_postcalibrate)]
        result = result.reset_index(drop = True)
        result[l] = np.abs(result[l] - y_train_interp_repeat)
        result = result.groupby("sample_id").mean()
        error = result[l]

    # Compute histogram and plot bar chart
    h, b = np.histogram(error, density = False)
    __ = ax.bar(0.5 * (b[1:] + b[0:-1]), h + 0.1, width = 0.9 * (b[1] - b[0]), align = "center", alpha = 1.0)

    # Plot samples as rain droplets
    jitter = np.sqrt(num_samples) * (rng.random((num_samples, )) - 1.0)
    __ = ax.scatter(error, jitter, marker = ".", color = cmap[1, :])

    # Guide line
    xlim = ax.get_xlim()
    __ = ax.plot(xlim, [0.0, 0.0], color = "black", alpha = 0.2)
    __ = plt.setp(ax, xlim = xlim)

    # Axis labels
    __ = plt.setp(ax, title = model_state, xlabel = "Mean Absolute Error (MAE)", ylabel = "Count")

    # Y ticks
    yticks = ax.get_yticks()
    yticks = yticks[yticks >= 0.0]
    __ = ax.set_yticks(yticks)

fig.savefig("./figures/test_calibrate_interface_state_errors.png")


# %%
