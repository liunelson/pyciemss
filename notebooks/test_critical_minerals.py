# %%[markdown]
# # Test simulate Interface
#
# Run Simulate with a simple model

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import pyciemss
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%
with open('../../critical-minerals/data/models/full-model-configured.json', 'r') as f:
    j = json.load(f)
    model = template_model_from_amr_json(j)

# %%
# Simulate
start_time = 0.0
end_time = 1.0
logging_step_size = 1.0
num_samples = 1

results = pyciemss.sample(
    model, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time,
    solver_method = 'rk4',
    solver_options = {
        'step_size': 0.1
    }
)

# %%
results["data"].head()

# %%
# Plot results

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

# %%
# Save results["data"] as CSV file
results["data"].to_csv("./data/test_simulate_interface.csv")

# %%
# Sensitivity analysis

# Settings
d1 = results['data']
ooi = 'incident_cases' # outcome of interest
ooi_timepoint = 15.0
pois = ['beta', 'gamma'] # parameters of interest

# Format column names
ooi_ = f'{ooi}_observable_state'
pois_ = [f'persistent_{p}_param' for p in pois]

d1_ = d1[d1['timepoint_unknown'] == ooi_timepoint][[ooi_] + pois_]

# %%
# Estimate sensitivity score with linear regression over the 0-1 normalized parameters
poi_scores = []
for p in pois_:
    x, y = d1_[[p, ooi_]].sort_values(by = p).values.transpose()
    x = (x - y.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    coef = np.polyfit(x, y, 1)

    poi_scores.append(coef[0])

# Sort the scores in descending order of their absolute value
poi_scores = sorted(poi_scores, key = abs, reverse = True)

# %%
fig, axes = plt.subplots(1, 2, figsize = (8, 4))
for ax, p, s in zip(axes, pois_, poi_scores):
    x, y = d1_[[p, ooi_]].sort_values(by = p).values.transpose()
    x = (x - y.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    coef = np.polyfit(x, y, 1)
    __ = ax.plot(x, y, label = 'Data')
    __ = ax.plot(x, np.poly1d(coef)(x), label = 'Fit')
    __ = plt.setp(ax, xlabel = f'Normalized {p}', ylabel = ooi, title = f'Score = {s:.5f}')
    __ = ax.legend()

# %%
fig, ax = plt.subplots(1, 1, figsize = (4, 3))

__ = ax.plot(np.zeros((len(pois), )), [-1, 3], color = 'k', linewidth = 0.75)
__ = ax.barh(pois, poi_scores)

__ = plt.setp(ax, ylim = (-0.5, 1.5), xlabel = 'Sensitivity Score', ylabel = 'Model Parameters')
__ = ax.invert_yaxis()

# %%
# Scatter plot
fig, axes = plt.subplots(len(pois), len(pois), figsize = (8, 8))
fig.tight_layout()
fig.subplots_adjust(wspace = 0.05, hspace = 0.05)

for i, p_i in enumerate(pois_):
    y = d1_[p_i]
    for j, p_j in enumerate(pois_):
        x = d1_[p_j]
        __ = axes[i, j].scatter(x, y, c = d1_[ooi_], marker = '.', alpha = 0.3, cmap = 'RdBu_r')

        __ = plt.setp(axes[i, j], xlim = (x.min(), x.max()), ylim = (y.min(), y.max()))
        if j == 0:
            __ = plt.setp(axes[i, j], ylabel = p_i)
        if i == len(pois) - 1:
            __ = plt.setp(axes[i, j], xlabel = p_j)

        __ = plt.setp(axes[i, j], xlim = (x.min(), x.max()), ylim = (y.min(), y.max()))
        if j == 0:
            __ = plt.setp(axes[i, j], ylabel = p_i)
        else:
            axes[i, j].tick_params(axis = 'y', labelleft = False)
        if i == len(pois) - 1:
            __ = plt.setp(axes[i, j], xlabel = p_j)
        else:
            axes[i, j].tick_params(axis = 'x', labelbottom = False)

# %%
# Heat map
n = int(np.ceil(np.sqrt(num_samples)))

fig, axes = plt.subplots(len(pois), len(pois), figsize = (8, 8))
fig.tight_layout()
fig.subplots_adjust(wspace = 0.05, hspace = 0.05)

for i, p_i in enumerate(pois_):

    y = np.histogram_bin_edges(d1_[p_i], bins = n)
    ooi_y = np.digitize(d1_[p_i], y, right = False)

    for j, p_j in enumerate(pois_):
        
        x = np.histogram_bin_edges(d1_[p_j], bins = n)
        ooi_x = np.digitize(d1_[p_j], x, right = False)

        H = np.empty((n, n))
        for k in range(n):
            for l in range(n):
                H[k, l] = d1_[ooi_][(ooi_y == k) & (ooi_x == l)].median()
                
        X, Y = np.meshgrid(x, y)
        __ = axes[i, j].pcolormesh(X, Y, H, cmap = 'RdBu_r')

        __ = plt.setp(axes[i, j], xlim = (x[0], x[-1]), ylim = (y[0], y[-1]))
        if j == 0:
            __ = plt.setp(axes[i, j], ylabel = p_i)
        else:
            axes[i, j].tick_params(axis = 'y', labelleft = False)
        if i == len(pois) - 1:
            __ = plt.setp(axes[i, j], xlabel = p_j)
        else:
            axes[i, j].tick_params(axis = 'x', labelbottom = False)

# %%
