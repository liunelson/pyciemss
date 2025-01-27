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
dataset.insert(loc = 0, column = 'Timestamp', value = dataset.pop('timestamp'))
print(tabulate(dataset, headers = 'keys'))

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
num_iterations = 20
num_samples =  100
start_time = 0.0
end_time = len(dataset) + 28.0 + 1.0
logging_step_size = 0.5

# %%
# Mapping model outputs (observables) of each model to ensemble model outputs
def solution_mapping(model_solution: dict) -> dict:
    mapped_dict = {}
    mapped_dict["D"] = model_solution["Deaths"]
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
        dirichlet_alpha = torch.ones(1)
    )
    results['single_simulate_precalibrate'].append(r)

    # Step 2: calibrate each model
    r = pyciemss.ensemble_calibrate(
        model_paths_or_jsons = [p], 
        solution_mappings = [solution_mapping],
        data_path = dataset,
        data_mapping = data_mapping,
        num_iterations = num_iterations
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
        inferred_parameters = r['inferred_parameters']
    )
    results['single_simulate_postcalibrate'].append(rr)

# %%


