# %%[markdown]
# # Test Ensemble Interfaces
#
# Sabina's notebooks:
# * [18-month epi evaluation ensemble challenge](https://github.com/ciemss/program-milestones/tree/12-epi-ensemble-challenge/18-month-milestone/evaluation/Epi_Ensemble_Challenge)
# * [12-month epi evaluation ensemble challenge](https://github.com/ciemss/pyciemss/tree/sa-ensemble-eval/notebook/ensemble_eval_sa)

# %%
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tabulate import tabulate

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

num_iterations = 10
num_samples =  10
start_time = 0.0
logging_step_size = 1.0

data_mapping_single = {
    'Infected': 'Cumulative_cases',
    'Hospitalized': 'Cumulative_hosp',
    'Dead': 'deceased'
}

# %%
r = pyciemss.ensemble_calibrate(
    model_paths_or_jsons = [model_paths[0]], 
    solution_mappings = [lambda x: x],
    data_path = dataset,
    data_mapping = data_mapping_single,
    num_iterations = num_iterations
)

# %%







# %%

# %%
# Load models and datasets

# MODEL_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
# DATASET_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

MODEL_PATH = "./data/simulation-integration/data/models"
DATASET_PATH = "./data/simulation-integration/data/datasets"

model_paths = [
    os.path.join(MODEL_PATH, "SEIRHD_NPI_Type1_petrinet.json"),
    os.path.join(MODEL_PATH, "SEIRHD_NPI_Type2_petrinet.json")
]

dataset_paths = [
    os.path.join(DATASET_PATH, "SIR_data_case_hosp.csv"),
    os.path.join(DATASET_PATH, "traditional.csv")
]

# %%
def generate_summary_table(model) -> pandas.DataFrame:

    data = {"name": [t.name for t in model.templates]}
    for k in ("subject", "outcome", "controller"):
        data[k] = [getattr(t, k).name if hasattr(t, k) else None for t in model.templates]

    data["controllers"] = [[c.name for c in getattr(t, k)] if hasattr(t, "controllers") else None for t in model.templates]
    data["controller(s)"] = [i if j == None else j for i, j in zip(data["controller"], data["controllers"])]
    __ = data.pop("controller")
    __ = data.pop("controllers")

    data["rate_law"] = [t.rate_law for t in model.templates]
    data["interactor_rate_law"] = [t.get_interactor_rate_law() for t in model.templates]

    df = pandas.DataFrame(data)

    return df

# %%
for model_path in model_paths:
    with open(model_path, "r") as fp:
        tm = template_model_from_amr_json(json.load(fp))

    # GraphicalModel.for_jupyter(tm)
    print(f"{tm.annotations.name}")
    print(tabulate(generate_summary_table(tm)))
    print("\n")

# %%
start_time = 0.0
end_time = 28.0
logging_step_size = 1.0
num_samples = 5
solution_mappings = [
    lambda x: x,
    lambda x: x
]

# %%
result = pyciemss.ensemble_sample(
    model_paths,
    solution_mappings = solution_mappings,
    start_time = start_time,
    end_time = end_time,
    logging_step_size = logging_step_size,
    num_samples = num_samples,
    time_units = 'days',
    dirichlet_alpha = torch.tensor([1, 1, 1])
)

# %%
















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
