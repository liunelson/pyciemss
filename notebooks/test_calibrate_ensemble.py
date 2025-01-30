# %%[markdown]
# # Test Calibrate Ensemble

# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tabulate import tabulate
from tqdm import tqdm

import pyciemss
from pyciemss.integration_utils.result_processing import cdc_format
from mira.modeling.viz import GraphicalModel
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%
model_paths = [
    './data/training/SEIRHD Model.json',
    './data/training/SEIRHD vacc model for LA County t0 = 10_28_2021.json'
]

dataset_path = './data/training/dataset.csv'

# %%
dataset = pd.read_csv(dataset_path)

# %%
fig, ax = plt.subplots(1, 1, figsize = (8, 6))
for c in dataset.columns:
    if c in ('cases', 'deaths'):
        t = pd.to_datetime(dataset['date'])
        __ = ax.plot(t, dataset[c], label = c, marker = '.')
__ = plt.setp(ax, yscale = 'log', xlabel = 'Date', ylabel = 'Persons', title = 'Calibration Dataset')
ax.tick_params(axis = 'x', labelrotation = 45)
__ = ax.legend()

# %%
dataset.insert(loc = 0, column = 'Timestamp', value = dataset.pop('timestamp'))
dataset = dataset[['Timestamp', 'deaths']]

print(tabulate(dataset, headers = 'keys'))

# %%
num_iterations = 50
num_samples =  100
start_time = 0.0
end_time = len(dataset) + 28.0 + 1.0
logging_step_size = 1.0
solver_method = 'dopri5'

# %%
# Mapping model outputs (observables) of each model to ensemble model outputs
def solution_mapping(model_solution: dict) -> dict:
    mapped_dict = {}
    mapped_dict["Deaths"] = model_solution["D"]
    return mapped_dict

# Mapping from dataset features to model outputs (observables)
data_mapping = {
    'deaths': 'Deaths'
}

# %%
# Calibrate each model as a single-model ensemble

results = {}
results['single_simulate_precalibrate'] = []
results['single_calibrate'] = []
results['single_simulate_postcalibrate'] = []

for i, p in tqdm(enumerate(model_paths)):

    # Step 1: simulate each model before calibration
    r = pyciemss.ensemble_sample(
        model_paths_or_jsons = [p],
        solution_mappings = [solution_mapping],
        start_time = start_time,
        end_time = end_time,
        logging_step_size = logging_step_size,
        num_samples = num_samples,
        dirichlet_alpha = torch.ones(1),
        solver_method = solver_method
    )
    results['single_simulate_precalibrate'].append(r)

    # Step 2: calibrate each model
    r = pyciemss.ensemble_calibrate(
        model_paths_or_jsons = [p], 
        solution_mappings = [solution_mapping],
        data_path = dataset,
        data_mapping = data_mapping,
        num_iterations = num_iterations,
        solver_method = solver_method
    )
    results['single_calibrate'].append(r)

    # Simulate each model after calibration
    rr = pyciemss.ensemble_sample(
        model_paths_or_jsons = [p],
        solution_mappings = [solution_mapping],
        start_time = start_time,
        end_time = end_time,
        logging_step_size = logging_step_size,
        num_samples = num_samples,
        inferred_parameters = r['inferred_parameters'],
        solver_method = solver_method
    )
    results['single_simulate_postcalibrate'].append(rr)

# %%
# Repeat for ensemble model
#
# Step 1: simulate ensemble before calibration
results['ensemble_simulate_precalibrate'] = pyciemss.ensemble_sample(
    model_paths_or_jsons = model_paths,
    solution_mappings = [solution_mapping, solution_mapping],
    start_time = start_time,
    end_time = end_time,
    logging_step_size = logging_step_size,
    num_samples = num_samples,
    dirichlet_alpha = torch.tensor([1.0, 1.0])
)

# %%
# Step 2: calibrate ensemble
results['ensemble_calibrate'] = pyciemss.ensemble_calibrate(
    model_paths_or_jsons = model_paths, 
    solution_mappings = [solution_mapping, solution_mapping],
    data_path = dataset,
    data_mapping = data_mapping,
    num_iterations = num_iterations
)

# %%
# Step 3: simulate ensemble after calibration
results['ensemble_simulate_postcalibrate'] = pyciemss.ensemble_sample(
    model_paths_or_jsons = model_paths,
    solution_mappings = [solution_mapping, solution_mapping],
    start_time = start_time,
    end_time = end_time,
    logging_step_size = logging_step_size,
    num_samples = num_samples,
    inferred_parameters = results['ensemble_calibrate']['inferred_parameters']
)

# %%
fig, axes = plt.subplots(len(dataset.columns), len(model_paths) + 1, figsize = (10, 10))
for i, (v, c) in enumerate(data_mapping.items()):
    for j in range(len(model_paths) + 1):
        ax = axes[i, j]

        # Calibration data
        x = dataset['Timestamp']
        y = dataset[v]
        h0, = ax.plot(x, y, linestyle = ':', label = 'Calibration Dataset')

        # Single models
        if j < len(model_paths):

            # Pre-calibration
            r = results['single_simulate_precalibrate'][j]['data']
            r = r.groupby(['timepoint_id']).aggregate('mean').reset_index()
            x = r['timepoint_unknown']
            y = r[f'{c}_state']
            __  = ax.plot(x, y, label = 'Pre-calibration')

            # Post-calibration
            r = results['single_simulate_postcalibrate'][j]['data']
            r = r.groupby(['timepoint_id']).aggregate('mean').reset_index()
            x = r['timepoint_unknown']
            y = r[f'{c}_state']
            __  = ax.plot(x, y, label = 'Post-Calibration')

        # Ensemble model
        else:
            # Pre-calibration
            r = results['ensemble_simulate_precalibrate']['data']
            r = r.groupby(['timepoint_id']).aggregate('mean').reset_index()
            x = r['timepoint_unknown']
            y = r[f'{c}_state']
            h1,  = ax.plot(x, y, label = 'Pre-calibration')

            # Post-calibration
            r = results['ensemble_simulate_postcalibrate']['data']
            r = r.groupby(['timepoint_id']).aggregate('mean').reset_index()
            x = r['timepoint_unknown']
            y = r[f'{c}_state']
            h2,  = ax.plot(x, y, label = 'Post-Calibration')

        # Formatting
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        if i == 0:
            if j < len(model_paths):
                l = results['single_calibrate'][j]['loss']
                __ = plt.setp(ax, title = f'model_{j}\nLoss = {l:.1f}')
            else:
                l = results['ensemble_calibrate']['loss']
                __ = plt.setp(ax, title = f'Ensemble Model\nLoss = {l:.1f}')
        elif i == 2:
            __ = plt.setp(ax, xlabel = 'Timepoint')
        if i != 2:
            __ = ax.tick_params(labelbottom = False)
        if j == 0:
            __ = plt.setp(ax, ylabel = c)

__ = fig.legend([h0, h1, h2], ['Calibration Dataset',  'Pre-Calibration',  'Post-Calibration'], loc = 'lower center', ncols = 3)

# %%
