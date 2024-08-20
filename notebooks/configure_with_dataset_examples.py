# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
import json
import re
import copy
import numpy as np
import pandas as pd
import pyciemss
import mira
from mira.sources.amr.petrinet import template_model_from_amr_json
from mira.modeling.amr.petrinet import template_model_to_petrinet_json

# %%[markdown]
# # Generate Configure with Dataset Examples
# 
# ##  1. Time-series Example
# 
# * Start with a configured SIDARTHE model
# * Run `pyciemss.sample` to get DataFrame CSV output

# %%
with open("./data/monthly_demo_202408/model_sidarthe_with_observable.json", "r") as f:
    amr_sidarthe = json.load(f)

# %%
start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 10

results = pyciemss.sample(
    amr_sidarthe, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time
)

# %%
# Normalize the column names
names_map = {}

for n in results["data"].columns:
    if re.search(r"[\w]+_observable_state", n):
        names_map[n] = re.search(r"[\w]+(?=_observable_state)", n).group(0)
    elif re.search(r"[\w]+_state", n):
        names_map[n] = re.search(r"[\w]+(?=_state)", n).group(0)
    elif re.search(r"persistent_[\w]+_param", n):
        names_map[n] = re.search(r"(?<=persistent_)[\w]+(?=_param)", n).group(0)
    else:
        pass


df = results["data"].rename(columns = names_map)
df.to_csv("./data/configure_with_dataset_examples/timeseries_example_dataset.csv", index = False)


# %%[markdown]
# ## 2. Stratified Parameter Example
# 
# * Start with a SEIRHD model
# * Stratify it with 18 age groups

# %%
with open("./data/monthly_demo_202408/model_seirhd.json", "r") as f:
    amr = json.load(f)
    model_seirhd = template_model_from_amr_json(amr)

# %%
num_age = 18

ages = [f'{i*5}_{(i+1)*5-1}' for i in range(17)] + ['85']

# %%
model_seirhd_age = mira.metamodel.stratify(
    model_seirhd,
    key = "age",
    strata = [f"{i}" for i in range(num_age)],
    structure = [],
    directed = False, 
    concepts_to_stratify = ["S", "I"],
    params_to_stratify = ["b"],
    cartesian_control = True
)

amr_seirhd_age = template_model_to_petrinet_json(model_seirhd_age)

# %%
with open("./data/configure_with_dataset_examples/model_seirhd_age.json", "w") as f:
    json.dump(template_model_to_petrinet_json(model_seirhd_age), f, indent = 4)

# %%
# Build subject-controller parameter matrix with parameter IDs
# for parameter `b`

params = np.asarray([f"b_{i}" for i in range(num_age ** 2)]).reshape((num_age, num_age))
subjects = [f"S_{i}" for i in range(num_age)]
controllers = [f"I_{i}" for i in range(num_age)]

# parameter matrix, model mapping
df = pd.DataFrame(params, index = subjects, columns = controllers)
df.to_csv("./data/configure_with_dataset_examples/model_mapping_example.csv", index = True)

# %%
# Dataset
df_data = pd.read_csv("./data/configure_with_dataset_examples/Contact Matrix_NY.csv", index_col = 0)

df_data = df_data.rename(
    index = {n1: n2 for n1, n2 in zip(df_data.index, df.index)}, 
    columns = {n1: n2 for n1, n2 in zip(df_data.columns, df.columns)}, 
)

df_data.to_csv("./data/configure_with_dataset_examples/dataset_example.csv")

# %%
parameter_data_map = {
    df.iloc[i, j]: float(df_data.iloc[i, j])
    for i in range(num_age) 
    for j in range(num_age)
}

model_seirhd_age_configured = copy.deepcopy(model_seirhd_age)

for k, v in parameter_data_map.items():
    model_seirhd_age_configured.parameters[k].value = v

# %%
with open("./data/configure_with_dataset_examples/model_seirhd_age_configured.json", "w") as f:
    json.dump(template_model_to_petrinet_json(model_seirhd_age_configured), f, indent = 4)

# %%

