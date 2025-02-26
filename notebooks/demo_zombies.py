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
from tqdm import tqdm
import torch

import pyciemss
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%
PATH = "./data/demo_zombies"

# %%
# Load models
models = {}
for p in sorted(os.listdir(PATH)):
    if os.path.isfile(os.path.join(PATH, p)):
        with open(os.path.join(PATH, p)) as fp:
            models[p.split('.')[0]] = template_model_from_amr_json(json.load(fp))

# %%
# Check configuration
def generate_config_table(model) -> pd.DataFrame:
    params = {f'{k}(t = 0)': float(str(v.expression)) for k, v in sorted(model.initials.items())}
    params = params | {k: v.value for k, v in model.parameters.items()}
    return pd.DataFrame({'Name': params.keys(), 'Value': params.values()})

# %%
# Model 2 (basic) - Fig 3 config
generate_config_table(models['Model 2 (SZR, basic)'])

# %%
# Model 4 (with quarantine)
generate_config_table(models['Model 4 (SIZRQ, with quarantine)'])

# %%
# Model 5 (with treatment)
generate_config_table(models['Model 5 (SIZR with treatment)'])

# %%
# Model 6 (with impulsive eradication)
generate_config_table(models['Model 6 (SZR, impulsive eradication)'])

# %%
# For Model 6
impulsive_eradication_policy = {
    2.5: {'n': torch.tensor(1.0)},
    2.6: {'n': torch.tensor(0.0)},
    5.0: {'n': torch.tensor(2.0)},
    5.1: {'n': torch.tensor(0.0)},
    7.5: {'n': torch.tensor(3.0)},
    7.6: {'n': torch.tensor(0.0)},
    10.0: {'n': torch.tensor(4.0)},
    10.1: {'n': torch.tensor(0.0)}
}

# %%
# Simulate
start_time = 0.0
end_time = 10.5
logging_step_size = 0.1
num_samples = 1

results = {}
for name, model in tqdm(models.items()):
    if name == 'Model 6 (SZR, impulsive eradication)':
        results[name] = pyciemss.sample(
            model, 
            end_time, 
            logging_step_size, 
            num_samples, 
            start_time = start_time,
            static_parameter_interventions = impulsive_eradication_policy,
        )['data']

    else:
        results[name] = pyciemss.sample(
            model, 
            end_time, 
            logging_step_size, 
            num_samples, 
            start_time = start_time
        )['data']


# %%
results[name].head()

# %%
# Plot results

# colors = mpl.colormaps["tab10"](range(10))
fig, ax = plt.subplots(len(results), 1, figsize = (8, 15))
for i, (name, r) in enumerate(results.items()):
    time = r['timepoint_unknown']
    for c in list(r.columns):
        # if c.split('_')[-1] == 'state':
        if c.split('_')[0] == 'Z':
            fig.axes[i].plot(time, r[c], label = c)
    fig.axes[i].legend()
    __ = plt.setp(fig.axes[i], xlim = (time.iloc[0], time.iloc[-1]), title = name)

    if i < (len(results) - 1):
        fig.axes[i].tick_params(axis = 'x', labelbottom = False)


r = results['Model 6 (SZR, impulsive eradication)']
fig, ax = plt.subplots(1, 1, figsize = (8, 3))
__ = ax.plot(time, r['persistent_n_param'], label = 'n')
__ = plt.setp(ax, xlim = (time.iloc[0], time.iloc[-1]))
__ = ax.legend()

# %%