# %%[markdown]
# # Test Calibrate Interface
#
# Run Simulate with a simple model

# %%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests

import pyciemss
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%
# Get test model (basic SIR)
PATH = "./data"
model_url = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/SIR_param_in_observables.json"
r = requests.get(model_url)
if r.ok:
    model = r.json()

# Modify the model configuration 
# by giving priors to parameter values

# beta
model["semantics"]["ode"]["parameters"][0] = {
    'id': 'beta',
    'value': 0.0025,
    'distribution': {
        'type': 'StandardUniform1',
        'parameters': {
            'minimum': 0.001, 
            'maximum': 0.003
        }
    }
}

# gamma
model["semantics"]["ode"]["parameters"][1] = {
    'id': 'gamma',
    'value': 0.07,
    'distribution': {
        'type': 'StandardUniform1',
        'parameters': {
            'minimum': 0.04,
            'maximum': 0.15
        }
    },
    'units': {
        'expression': '1/day',
        'expression_mathml': '<apply><power/><ci>day</ci><cn>-1</cn></apply>'
    }
}

# Save modified model
with open(os.path.join(PATH, "test_simulate_model.json"), "w") as fp:
    json.dump(model, fp, indent = 4)


# %%
# Simulate
start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 10

results = pyciemss.sample(
    model, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time
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
