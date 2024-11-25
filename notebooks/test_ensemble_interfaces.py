# %%[markdown]
# # Test Ensemble Interfaces
#
# Sabina's notebooks:
# * [18-month epi evaluation ensemble challenge](https://github.com/ciemss/program-milestones/tree/12-epi-ensemble-challenge/18-month-milestone/evaluation/Epi_Ensemble_Challenge)
# * [12-month epi evaluation ensemble challenge](https://github.com/ciemss/pyciemss/tree/sa-ensemble-eval/notebook/ensemble_eval_sa)

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
MODEL_PATH = "./data/18m_eval/models"
DATASET_PATH = "./data/18m_eval/datasets"

# %%
# Models
model_paths = [
    os.path.join(MODEL_PATH, 'SEIRHD_age_structured_petrinet.json'),
    os.path.join(MODEL_PATH, 'SEIRHD_vacc_var_petrinet.json'),
    os.path.join(MODEL_PATH, 'SEIRHD_base_petrinet.json'),
]

# %%
# Select calibration dataset

location = 'New York State'
start_date = '2021-06-01'
end_date = '2021-09-06'

full_dataset = pd.read_csv(os.path.join(DATASET_PATH, 'full_dataset.csv'))
full_dataset = full_dataset.sort_values(by = 'date').reset_index(drop = True)
dataset = full_dataset[(full_dataset.date >= start_date) & (full_dataset.date < end_date)].reset_index(drop = True).reset_index(names = ['Timestamp'])

# Drop date because pyciemss rejects them
dataset = dataset.drop(labels = 'date', axis = 1)

print(tabulate(dataset, headers = 'keys'))

# %%
fig, ax = plt.subplots(1, 1, figsize = (4, 3))
for c in dataset.columns:
    if c not in ('Timestamp', 'date'):
        __ = ax.plot(dataset['Timestamp'], dataset[c], label = c)
__ = plt.setp(ax, yscale = 'log', xlabel = 'Timestamp', ylabel = 'Persons', title = 'Calibration Dataset')
__ = ax.legend()


# %%
# Settings

num_iterations = 20
num_samples =  200
start_time = 0.0
end_time = len(dataset) + 28.0 + 1.0
logging_step_size = 1.0

# %%
# Mapping model outputs (observables) of each model to ensemble model outputs
def solution_mapping(model_solution: dict) -> dict:
    mapped_dict = {}
    mapped_dict["Susceptible"] = model_solution["susceptible"]
    mapped_dict["Exposed"] = model_solution["exposed"]
    mapped_dict["Infected"] = model_solution["infected"]
    mapped_dict["Recovered"] = model_solution["recovered"]
    mapped_dict["Hospitalized"] = model_solution["hospitalized"]
    mapped_dict["Deceased"] = model_solution["deceased"]
    mapped_dict["Cumulative_cases"] = model_solution["cumulative_cases"]
    mapped_dict["Cumulative_hosp"] = model_solution["cumulative_hosp"]
    return mapped_dict

# %%
# Mapping from dataset features to model outputs (observables)
data_mapping = {
    'Infected': 'Cumulative_cases',
    'Hospitalized': 'Cumulative_hosp',
    'Dead': 'Deceased'
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
# Repeat for ensemble model
#
# Step 1: simulate ensemble before calibration
results['ensemble_simulate_precalibrate'] = pyciemss.ensemble_sample(
    model_paths_or_jsons = model_paths,
    solution_mappings = [solution_mapping, solution_mapping, solution_mapping],
    start_time = start_time,
    end_time = end_time,
    logging_step_size = logging_step_size,
    num_samples = num_samples,
    dirichlet_alpha = torch.tensor([1.0, 1.0, 1.0])
)

# %%
# Step 2: calibrate ensemble
results['ensemble_calibrate'] = pyciemss.ensemble_calibrate(
    model_paths_or_jsons = model_paths, 
    solution_mappings = [solution_mapping, solution_mapping, solution_mapping],
    data_path = dataset,
    data_mapping = data_mapping,
    num_iterations = num_iterations
)

# %%
# Step 3: simulate ensemble after calibration
results['ensemble_simulate_postcalibrate'] = pyciemss.ensemble_sample(
    model_paths_or_jsons = model_paths,
    solution_mappings = [solution_mapping, solution_mapping, solution_mapping],
    start_time = start_time,
    end_time = end_time,
    logging_step_size = logging_step_size,
    num_samples = num_samples,
    inferred_parameters = results['ensemble_calibrate']['inferred_parameters']
)

# %%
# Plot results

fig, axes = plt.subplots(3, 4, figsize = (10, 10))
for i, (v, c) in enumerate(data_mapping.items()):
    for j in range(4):
        ax = axes[i, j]

        # Calibration data
        x = dataset['Timestamp']
        y = dataset[v]
        h0, = ax.plot(x, y, linestyle = ':', label = 'Calibration Dataset')

        # Single models
        if j < 3:

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
            if j < 3:
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
# Plot prior/posterior distributions

# Pick a single model and one of its parameters
j = 0
param = 'beta'

fig, axes = plt.subplots(1, 2, figsize = (10, 4))

# Single model
ax = axes[0]

# Before calibration
r = results['single_simulate_precalibrate'][j]['data'].groupby(['sample_id']).aggregate('mean').reset_index()
d = r[f'model_{j}/persistent_{param}_param']
c, b = np.histogram(d, bins = int(np.sqrt(num_samples)))
x = 0.5 * (b[1:] + b[:-1])
w = 0.9 * (b[1] - b[0])
h0 = ax.bar(x, c, width = w, align = 'center', alpha = 0.5, label = 'Pre-calibration')

# After calibration
r = results['single_simulate_postcalibrate'][j]['data'].groupby(['sample_id']).aggregate('mean').reset_index()
d = r[f'model_{j}/persistent_{param}_param']
c, b = np.histogram(d, bins = b)
x = 0.5 * (b[1:] + b[:-1])
w = 0.9 * (b[1] - b[0])
h1 = ax.bar(x, c, width = w, align = 'center', alpha = 0.5, label = 'Post-calibration')


# Ensemble model
ax = axes[1]

# Before calibration
r = results['ensemble_simulate_precalibrate']['data'].groupby(['sample_id']).aggregate('mean').reset_index()
d = r[f'model_{j}/persistent_{param}_param']
c, b = np.histogram(d, bins = int(np.sqrt(num_samples)))
x = 0.5 * (b[1:] + b[:-1])
w = 0.9 * (b[1] - b[0])
__ = ax.bar(x, c, width = w, align = 'center', alpha = 0.5, label = 'Post-calibration')

# After calibration
r = results['ensemble_simulate_postcalibrate']['data'].groupby(['sample_id']).aggregate('mean').reset_index()
d = r[f'model_{j}/persistent_{param}_param']
c, b = np.histogram(d, bins = b)
x = 0.5 * (b[1:] + b[:-1])
w = 0.9 * (b[1] - b[0])
h1 = ax.bar(x, c, width = w, align = 'center', alpha = 0.5, label = 'Post-calibration')

axes[0].ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0))
axes[1].ticklabel_format(axis = 'x', style = 'sci', scilimits = (0, 0))
__ = plt.setp(axes[0], title = f'Model {j}\n as Single Model', xlabel = f'Values of the Parameter {param}', ylabel = 'Sample Counts')
__ = plt.setp(axes[1], title = f'Model {j}\n as Part of Ensemble Model', xlabel = f'Values of the Parameter {param}')
__ = ax.legend([h0, h1], ['Pre-Calibration', 'Post-Calibration'], loc = 'best', ncols = 1)


# %%
# Model weight distributions

fig, axes = plt.subplots(1, 3, figsize = (10, 4))
for j, __ in enumerate(model_paths):
    ax = axes[j]

    # Before calibration
    r = results['ensemble_simulate_precalibrate']['data'].groupby(['sample_id']).aggregate('mean').reset_index()
    c, b = np.histogram(r[f'model_{j}/weight_param'], bins = int(np.sqrt(num_samples)), range = (0, 1))
    x = 0.5 * (b[1:] + b[:-1])
    w = 0.9 * (b[1] - b[0])
    __ = ax.bar(x, c, width = w, align = 'center', alpha = 0.5)

    # After calibration
    r = results['ensemble_simulate_postcalibrate']['data'].groupby(['sample_id']).aggregate('mean').reset_index()
    c, b = np.histogram(r[f'model_{j}/weight_param'], bins = int(np.sqrt(num_samples)), range = (0, 1))
    x = 0.5 * (b[1:] + b[:-1])
    w = 0.9 * (b[1] - b[0])
    __ = ax.bar(x, c, width = w, align = 'center', alpha = 0.5)


    __ = plt.setp(ax, xlabel = f'Weight of Model {j}\nin the Ensemble Model')
    if j == 0:
        __ = plt.setp(ax, ylabel = 'Sample Count')

# %%
# CDC Data Formatting

ensemble_dataset = cdc_format(
    results['ensemble_simulate_postcalibrate']['ensemble_quantiles'],
    # solution_string_mapping = {v: v for k, v in data_mapping.items()},
    solution_string_mapping = {
        'Deceased_state': 'cum death',
        'model_0/deceased_observable_state': 'cum death 0',
        'model_1/deceased_observable_state': 'cum death 1',
        'model_2/deceased_observable_state': 'cum death 2'
    },
    forecast_start_date = end_date,
    location = location,
    drop_column_names = ['timepoint_id', 'number_days'],
    train_end_point = len(dataset) - 1.0
)
ensemble_dataset = ensemble_dataset.reset_index(drop = True)

print(tabulate(ensemble_dataset.head(10), headers = 'keys'))

# %%
# Plot CDC data

fig, ax = plt.subplots(1, 1, figsize = (10, 8))

# Calibrated Ensemble Forecast

outputs = ensemble_dataset['output'].unique()
quantiles =ensemble_dataset['quantile'].unique()
num_quantiles = len(quantiles)

for output in outputs:

    r = ensemble_dataset[ensemble_dataset['output'] == output]
    
    if output == 'cum death':
        cmap = 'Greens'
    elif output == 'cum death 0':
        cmap = 'Reds'
    elif output == 'cum death 1':
        cmap = 'Blues'
    elif output == 'cum death 2':
        cmap = 'Purples'

    colors = [mpl.colormaps.get_cmap(f'{cmap}')(i) for i in np.linspace(0, 1, int(0.5 * (num_quantiles + 1)))]
    colors += [mpl.colormaps.get_cmap(f'{cmap}_r')(i) for i in np.linspace(0, 1, int(0.5 * (num_quantiles + 1)))]

    for i, q in enumerate(quantiles):
        c = colors[i]
        rr = r[r['quantile'] == q].sort_values('target_end_date')

        x = rr['target_end_date']
        y = rr['value']
        __ = ax.plot(x, y, color = c, label = f'{q}')


# Calibration dataset
x = np.arange(np.datetime64(start_date), np.datetime64(end_date), np.timedelta64(1, 'D'))
y = dataset['Dead']
__ = ax.plot(x, y, linestyle = ':', label = 'Calibration Dataset', color = 'k')

# Formatting
ax.tick_params('x', labelrotation = 45.0)
__ = plt.setp(ax, ylabel = 'Persons', title = 'Quantiles of "cum death" Forecast from Ensemble Model')
 
__ = ax.legend(
    [
        mpl.lines.Line2D([0], [0], color = 'k', linestyle = ':'),
        mpl.lines.Line2D([0], [0], color = 'green', linestyle = '-'),
        mpl.lines.Line2D([0], [0], color = 'red', linestyle = '-'),
        mpl.lines.Line2D([0], [0], color = 'blue', linestyle = '-'),
        mpl.lines.Line2D([0], [0], color = 'purple', linestyle = '-')
    ], 
    ['Calibration Data', 'Ensemble', 'Model A', 'Model B', 'Model C']
)

# %%
