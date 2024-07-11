# %%[markdown]
# # Test Calibrate Interface
#
# Run Calibrate and do exploratory analysis of the results.


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
# Calibrate Settings

start_time = 0.0
end_time = 5.0
logging_step_size = 0.1
num_samples = 100

# %%
# Run pre-calibration Simulate

result_precalibrate = pyciemss.sample(model1, end_time, logging_step_size, num_samples, start_time=start_time)

result_precalibrate["data"].head()

# %%
# Run Calibrate

data_mapping = {"case": "infected", "hosp": "hospitalized"} # data is mapped to observables
# data_mapping = {"case": "I", "hosp": "H"} # data is mapped to state variables

num_iterations = 10

result_calibrate = pyciemss.calibrate(
    model1, 
    dataset1, 
    data_mapping = data_mapping, 
    num_iterations = num_iterations
)

# %%
result_calibrate

# result_calibrate = {
#     "inferred_parameters": AutoGuideList((0): AutoDelta(), (1): AutoLowRankMultivariateNormal())
#     "loss": <float>y
# }

# %%
parameter_estimates = result_calibrate["inferred_parameters"]
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
# Run post-calibration Simulate
result_postcalibrate = pyciemss.sample(
    model1, 
    end_time, 
    logging_step_size, 
    num_samples,
    start_time = start_time, 
    inferred_parameters = result_calibrate["inferred_parameters"]
)

result_postcalibrate["data"].head()

# %%
# Plot trajectories

cmap = mpl.colormaps["tab10"](range(10))
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))
fig.suptitle("State Variable Trajectories")
fig.tight_layout()

for ax, (train_data_state, model_state) in zip(axes, data_mapping.items()):

    # Pre-/post-calibrate trajectories
    for result_, label, color in zip((result_precalibrate, result_postcalibrate), ("Pre-Calibration", "Post-Calibration"), (cmap[0, :], cmap[1, :])):

        # Timepoints
        x = result_["data"][result_["data"]["sample_id"] == 0]["timepoint_unknown"]

        for i in range(num_samples):
            result = result_["data"][result_["data"]["sample_id"] == i]
            
            if "_".join([model_state, "state"]) in result.columns:
                l = "_".join([model_state, "state"])
                y = result[l]
            elif "_".join([model_state, "observable", "state"]) in result.columns:
                l = "_".join([model_state, "observable", "state"])
                y = result[l]
            else:
                l = None
                y = None
            
            if ~isinstance(y, type(None)):
                ax.plot(x, y, linewidth = 0.5, color = color, alpha = 0.8, label = label)


    # Training dataset trajectories
    ax.plot(dataset1_df["timepoint"], dataset1_df[train_data_state], marker = "o", color = "black", markeredgecolor = "black", markerfacecolor= "white", label = train_data_state)      
    
    # Axis labels
    __ = plt.setp(ax, xlabel = "Time (Days)", ylabel = model_state)

    # Legend
    l = [
        mpl.lines.Line2D([0], [0], label = "Observations", marker = "o", color = "black", markeredgecolor = "black", markerfacecolor= "white"),
        mpl.lines.Line2D([0], [0], label = "Pre-Calibration", linewidth = 0.5, color = cmap[0, :], alpha = 0.8),
        mpl.lines.Line2D([0], [0], label = "Post-Calibration", linewidth = 0.5, color = cmap[1, :], alpha = 0.8)
    ]
    __ = ax.legend(handles = l)


fig.savefig("./figures/test_calibrate_interface_state_trajectories.png")

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
