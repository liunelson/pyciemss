# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# Review Integration Interfaces:
#
# 1. Optimize

# %%
import os
import numpy as np
import torch
from pyciemss.ouu.qoi import obs_nday_average_qoi

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# %%
# import pandas as pd
# import numpy as np
# import scipy as sp
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import json
# from typing import NoReturn, Optional, Any
# import itertools
# from tqdm import tqdm

# import os
# from pyciemss.PetriNetODE.interfaces_bigbox import (
#     load_and_sample_petri_model,
#     load_and_optimize_and_sample_petri_model,
#     load_and_calibrate_and_sample_petri_model,
#     load_and_calibrate_and_optimize_and_sample_petri_model,
    
# )
# from pyciemss.PetriNetODE.interfaces import get_posterior_density_mesh_petri

# %%[markdown]
# ## 1. Optimize

# %%

MODELS_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"


os.path.join(MODELS_PATH, "SIR_stockflow.json")

optimize_kwargs = {
    "qoi": lambda x: obs_nday_average_qoi(x, ["I_state"], 1),
    "risk_bound": 300.0,
    "static_parameter_interventions": {torch.tensor(1.0): "p_cbeta"},
    "objfun": lambda x: np.abs(0.35 - x),
    "initial_guess_interventions": 0.15,
    "bounds_interventions": [[0.1], [0.5]],
}



# %%
model_path = "BIOMD0000000955_askenet.json"
# model_path = "../../../test/models/AMR_examples/SIDARTHE.amr.json"
# model_path = "../../../test/models/AMR_examples/BIOMD0000000955_askenet.json"
# model_path = "./sir_typed.json"
with open(model_path, 'r') as f:
    model = json.load(f)

num_samples = 10
timepoints = np.arange(100, dtype = float)

start_time = timepoints[0] - 1e-5

# Parse initial condition
# start_state = {var['target']: float(var['expression']) for var in model['semantics']['ode']['initials']}
start_state = {}
for var in model['semantics']['ode']['initials']:
    start_state[var['target']] = var['expression']
    for param in model['semantics']['ode']['parameters']:
        start_state[var['target']] = start_state[var['target']].replace(param['id'], str(param['value']))
    start_state[var['target']] = float(start_state[var['target']])

# %%
# Run model simulation
results = load_and_sample_petri_model(
    model_path, 
    num_samples, 
    timepoints = timepoints, 
    # add_uncertainty = True, # temporarily removed due to bug
    start_time = start_time,
    start_state = start_state
)

results["data"].to_csv('./simulation_results.csv', index = False)

# %%
# Inference
def plot_results(results: pd.DataFrame) -> NoReturn:

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    for k, h in enumerate(results["data"].columns):
        if h.split("_")[-1] in ("sol",):
            x = results["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).mean().index.values
            y = results["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).mean()[h].values
            yerr = results["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).sem()[h].values
            __ = ax.errorbar(x, y, yerr = yerr, label =  h)
    __ = plt.setp(ax, xlabel = 'Time', ylabel = 'State Variables')
    __ = ax.legend(loc = "upper left")


    # Parameter distribution
    parameters = [c for c in results["data"].columns if c.split('_')[-1] == "param"]
    n = int(np.ceil(np.sqrt(len(parameters))))
    fig, axes = plt.subplots(n, n, figsize = (8, 8))
    fig.tight_layout()
    for p, ax in zip(parameters, fig.axes):
        __ = ax.hist(results["data"][results["data"]["timepoint_id"] == 0][p], label = f"{p.split('_')[0]}")
        __ = ax.legend(loc = "upper left")
        __ = plt.setp(ax, ylim = (0, len(np.unique(results["data"]["sample_id"]))))


plot_results(results)

# %%[markdown]
# results.data: dataframe with simulation result
# results.quantiles: dataframe with simmulation results in CDC quantile format
# results.risk: dict of dict, one for each state variable, containing a "risk" and "qoi"
# 
# The "qoi" (= Quantity of Interest) is the 2-day average of the state at the final timepoint

# %%[markdown]
# ## 2. Calibrate-Simulate

# %%
# Raw Forecast Hub dataset
df = pd.read_csv('./forecast_hub_raw_incident_data.csv')

# Region of interest
# 50 - 80
train_data = pd.DataFrame({
    'Timestep': np.arange(30, dtype = float),
    # 'Susceptible': (329.5e6 - np.array(df['cases'].iloc[50:80].cumsum())) / 329.5e6,
    'Extinct': np.array(df['deaths'].iloc[50:80]) / 329.5e6
})

train_data.to_csv('./train_data.csv', index = False)

# %%
# Run model calibration
data_path = './train_data.csv'
num_samples = 100
timepoints = np.arange(30, dtype = float)
num_iterations = 5

results_calibrated = load_and_calibrate_and_sample_petri_model(
    model_path,
    data_path,
    num_samples,
    timepoints = timepoints, 
    start_state = start_state,
    # add_uncertainty = True, 
    verbose = True,
    num_iterations = num_iterations,
)

results_calibrated["data"].to_csv('./calibrated_results.csv', index = False)

# %%
# results_calibrated
#   data: dataframe with simulation result
#   quantiles: dataframe with simmulation results in CDC quantile format
#   risk: dict of dict, one for each state variable, containing a "risk" and "qoi"
#   inferred_parameters: Pyro object that contains the joint posterior distribution over parameter space (i.e. the result of calibration)

# %%
plot_results(results_calibrated)

# %%
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
__ = ax.plot(train_data["Timestep"], train_data["Extinct"], marker = 'o', mfc = 'None', label = 'Training Data')
for k, h in enumerate(results["data"].columns):
    if h == "Extinct_sol":
        x = results_calibrated["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).mean().index.values
        y = results_calibrated["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).mean()[h].values
        yerr = results_calibrated["data"][["timepoint_id", "sample_id", h]].groupby(["timepoint_id"]).sem()[h].values
        __ = ax.errorbar(x, y, yerr = yerr, label =  h)
__ = ax.legend()

# %%
# Inspect `inferred_parameters`

# %%
# 2D approach

# Get parameter mesh ranges
n = 10
p1 = "zeta"
p2 = "lambda"
mesh_params = {
    "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
    if "_".join(c.split("_")[:-1]) in (p1, p2)
    else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
    for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
}

# Evaluate probability density over given parameter mesh
params, density = get_posterior_density_mesh_petri(
    inferred_parameters = results_calibrated["inferred_parameters"],
    mesh_params = mesh_params
)

# Marginalize all other parameters
axes = tuple([i for i, p in enumerate(params) if p not in (p1, p2)])
marginal_density = density.logsumexp(axis = axes)
# marginal_density = density.sum(axis = axes)
s = tuple([slice(None) if p in (p1, p2) else slice(0, 1) for p in params.keys()])
x = params[p1][s].squeeze()
y = params[p2][s].squeeze()

# Plot marginal distribution
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
i = ax.pcolormesh(x, y, marginal_density)
__ = plt.setp(ax, xlabel = p1, ylabel = p2, title = f"Marginalized Parameter Distribution from Model Calibration")
__ = fig.colorbar(i, ax = ax)

x0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p1 + "_param"]
y0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p2 + "_param"]
__ = ax.scatter(x0, y0, marker = '.', color = 'red')

# %%
# 3D approach

# Get parameter mesh ranges
n = 10
p1 = "beta"
p2 = "delta"
p3 = "alpha"
mesh_params = {
    "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
    if "_".join(c.split("_")[:-1]) in (p1, p2, p3)
    else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
    for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
}

# Evaluate probability density over given parameter mesh
params, density = get_posterior_density_mesh_petri(
    inferred_parameters = results_calibrated["inferred_parameters"],
    mesh_params = mesh_params
)

# Marginalize all other parameters
axes = tuple([i for i, p in enumerate(params) if p not in (p1, p2, p3)])
marginal_density = density.sum(axis = axes)
s = tuple([slice(None) if p in (p1, p2, p3) else slice(0, 1) for p in params.keys()])
x = params[p1][s].squeeze()
y = params[p2][s].squeeze()
z = params[p3][s].squeeze()

# %%
# Plot marginal distribution
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(projection = "3d")

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin = marginal_density.min().item(), vmax = marginal_density.max().item())
mappable = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
fig.colorbar(
    mappable, 
    cax = None, ax = ax, 
    # location = "bottom", orientation = "horizontal", 
    location = "right", orientation = "vertical", 
    label = "P", 
    fraction = 0.03,
    shrink = 1.0
)

# fc = np.zeros(marginal_density.shape + (3, ), dtype = float)
# n, __, __ = marginal_density.shape
# for i in range(n):
#     for j in range(n):
#         for k in range(n):
#             c = np.array(cmap(norm(marginal_density[i, j, k])))
#             c[-1] = 0.2 # alpha
#             c = c[:-1]
#             fc[i, j, k, :] = c
#
# h = ax.voxels(
#     x.numpy(), 
#     y.numpy(), 
#     z.numpy(), 
#     filled = marginal_density[:-1, :-1, :-1] > marginal_density.quantile(0.5), 
#     # facecolor = fc[:-1, :-1, :-1], 
#     # edgecolor = fc[:-1, :-1, :-1],
#     # shade = False
# )

for k, l in enumerate(z[0, 0, :]):
    h = ax.contour(
        x[:, :, k], y[:, :, k], 
        marginal_density[:, :, k], 
        zdir = 'z', offset = l, 
        levels = [marginal_density.quantile(v) for v in [0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]],
        cmap = cmap, norm = norm, alpha = 0.5
    )

__ = plt.setp(
    ax, 
    xlabel = p1, ylabel = p2, zlabel = p3, 
    xlim = mesh_params[p1][:2],
    ylim = mesh_params[p2][:2],
    zlim = mesh_params[p3][:2],
    title = f"Marginalized Parameter Distribution from Model Calibration"
)
ax.set_box_aspect((1, 1, 1))
# ax.set_aspect("equal")


# Sample points
x0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p1 + "_param"]
y0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p2 + "_param"]
z0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p3 + "_param"]
__ = ax.scatter(x0, y0, z0, marker = '.', color = 'red')

# %%
# Entropy of the marginal distributions (all pairwise combinations of parameters)
# relative to uniform distribution

list_params = ["_".join(c.split("_")[:-1]) for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"]

rel_entr = {}
for (p1, p2) in tqdm(itertools.combinations(list_params, 2)):

    n = 10
    mesh_params = {
        "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
        if "_".join(c.split("_")[:-1]) in (p1, p2)
        else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
        for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
    }

    # Evaluate probability density over given parameter mesh
    params, density = get_posterior_density_mesh_petri(
        inferred_parameters = results_calibrated["inferred_parameters"],
        mesh_params = mesh_params
    )

    # Marginalize all other parameters
    axes = tuple([i for i, p in enumerate(params) if p not in (p1, p2)])
    marginal_density = density.sum(axis = axes)

    rel_entr[(p1, p2)] = sp.special.rel_entr(marginal_density.numpy(), np.ones(marginal_density.shape) * marginal_density.mean().item()).sum()


# Sort parameter pairs by relative entropy
rel_entr = {k: rel_entr[k] for k in sorted(rel_entr, key = rel_entr.get, reverse = True)}

# %%
# Plot all pairwise marginal distributions
num_params = len(list_params)
fig, axes = plt.subplots(nrows = num_params, ncols = num_params, figsize = (12, 12))
cmap = mpl.cm.Greens
# norm = mpl.colors.Normalize(vmin = 0.0)
for i, p1 in enumerate(list_params):
    for j, p2 in enumerate(list_params):

        ax = axes[i, j]

        if j == 0:
            __ = plt.setp(ax, ylabel = f"{p1}")
        if i == num_params - 1:
            __ = plt.setp(ax, xlabel = f"{p2}")
        if (j != 0) and (i != num_params - 1):
            ax.tick_params('both', labelsize = 'xx-small', bottom = False, left = False, labelbottom = False, labelleft = False)
        else:
            ax.tick_params('both', labelsize = 'xx-small')

        if j < i:

            # Get parameter mesh ranges
            n = 5
            mesh_params = {
                "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
                if "_".join(c.split("_")[:-1]) in (p1, p2)
                else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
                for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
            }

            # Evaluate probability density over given parameter mesh
            params, density = get_posterior_density_mesh_petri(
                inferred_parameters = results_calibrated["inferred_parameters"],
                mesh_params = mesh_params
            )

            # Marginalize all other parameters
            axis = tuple([i for i, p in enumerate(params) if p not in (p1, p2)])
            marginal_density = density.sum(axis = axis)
            s = tuple([slice(None) if p in (p1, p2) else slice(0, 1) for p in params.keys()])
            x = params[p1][s].squeeze()
            y = params[p2][s].squeeze()

            # Plot marginal distribution
            h = ax.pcolormesh(x, y, marginal_density, cmap = cmap, vmin = 0.0)
        
        elif i == j:

            # Get parameter mesh ranges
            n = 5
            mesh_params = {
                "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
                if "_".join(c.split("_")[:-1]) in (p1, p2)
                else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
                for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
            }

            # Evaluate probability density over given parameter mesh
            params, density = get_posterior_density_mesh_petri(
                inferred_parameters = results_calibrated["inferred_parameters"],
                mesh_params = mesh_params
            )

            # Marginalize all other parameters
            axis = tuple([i for i, p in enumerate(params) if p not in (p1, p2)])
            marginal_density = density.sum(axis = axis)
            s = tuple([slice(None) if p in (p1, p2) else slice(0, 1) for p in params.keys()])
            x = params[p1][s].squeeze()
            # y = params[p2][s].squeeze()

            h = ax.bar(x, marginal_density, width = x[1] - x[0], bottom = 0.0, color = "green")

        else:
            ax.remove()

        ax.tick_params('both', labelsize = 'xx-small', bottom = False, left = False, labelbottom = False, labelleft = False)

# %%
# Plot top-16 pairwise marginal distributions sorted by relative entropy
fig, axes = plt.subplots(4, 4, figsize = (10, 10))
fig.subplots_adjust(wspace = 0.5, hspace = 0.35)
fig.suptitle(f"Top-{len(fig.axes)} Pairwise Marginal Distributions, Sorted by Relative Entropy")
cmap = mpl.cm.Blues
# norm = mpl.colors.Normalize(vmin = 0.0, vmax = )
for i, ((p1, p2), s) in enumerate(rel_entr.items()):

    if i < len(fig.axes):

        # Get parameter mesh ranges
        n = 10
        mesh_params = {
            "_".join(c.split("_")[:-1]): [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), n]
            if "_".join(c.split("_")[:-1]) in (p1, p2)
            else [results_calibrated["data"][c].min(), results_calibrated["data"][c].max(), 1]
            for c in results_calibrated["data"].columns if c.split("_")[-1] == "param"
        }

        # Evaluate probability density over given parameter mesh
        params, density = get_posterior_density_mesh_petri(
            inferred_parameters = results_calibrated["inferred_parameters"],
            mesh_params = mesh_params
        )

        # Marginalize all other parameters
        axis = tuple([i for i, p in enumerate(params) if p not in (p1, p2)])
        marginal_density = density.sum(axis = axis)
        s = tuple([slice(None) if p in (p1, p2) else slice(0, 1) for p in params.keys()])
        x = params[p1][s].squeeze()
        y = params[p2][s].squeeze()

        # Plot marginal distribution
        ax = fig.axes[i]
        h = ax.pcolormesh(x, y, marginal_density, cmap = cmap, vmin = 0.0, vmax = 1e-16)
        __ = plt.setp(ax, xlabel = p1, ylabel = p2, title = f"")
        ax.tick_params('both', labelsize = 'xx-small')

        x0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p1 + "_param"]
        y0 = results_calibrated["data"][results_calibrated["data"]["timepoint_id"] == 0][p2 + "_param"]
        # __ = ax.scatter(x0, y0, marker = '.', color = 'red', alpha = 0.3)

# __ = fig.colorbar(h, ax = ax)


# %%[markdown]
# ## 3. Optimize-Simulate

# %%
num_samples = 5
timepoints = np.arange(5.0, dtype = float)
start_time = timepoints[0] - 1e-5

# Interventions over the parameters
# interventions = [(2.1, "beta")]
# intervention_bounds = [[0.0], [3.0]]
interventions = [(2.5, "beta"), (3.2, "beta")]
intervention_bounds = [[0.0, 0.5], [3.0, 1.0]]

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
    start_state = start_state,
    n_samples_ouu = int(100),
    method = "euler",
    alpha_qs = [0.25, 0.5, 0.75, 0.95]
)

# Took 359.45 seconds

# %%
# optimize_result
#  data: dataframe
#  policy
#    policy: optimal value of intervention parameter
#    OptResults: scipy optimize result message
#    risk: alpha-superquantile risk value (alpha = 0.95)
#    samples: sample values of each model parameter
#    qoi: num_samples of qoi values
#  quantiles: dataframe

# %%
plot_results(optimize_result)

# %%[markdown]
# ## 4. Calibrate-Optimize-Simulate

# %%
optimize_calibrated_result = load_and_calibrate_and_optimize_and_sample_petri_model(
    model_path, 
    data_path,
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
    start_state = start_state
)

# %%
# Note: no "inferred_parameters" key in this result