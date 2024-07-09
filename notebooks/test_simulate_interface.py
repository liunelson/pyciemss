# %%[markdown]
# # Test Calibrate Interface
#
# Run Simulate with a simple model

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import pyciemss
from mira.sources.amr.petrinet import template_model_from_amr_json

# %%
start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 1

# %%
model_path = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/SIR_param_in_observables.json"

results = pyciemss.sample(
    model_path, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time
)

# %%
results["data"].head()

# %%

df = results["data"].groupby(["timepoint_id"]).mean()

fig, ax = plt.subplots(1, 1, figsize = (8, 6))

for c in results["data"].columns:

    if c.split("_")[-1] == "state":
        df = results["data"].groupby(["timepoint_id"]).mean()
        __ = ax.plot(df["timepoint_unknown"], df[c], label = c)

__ = ax.legend()

# %%
# Save results["data"] as CSV file
results["data"].to_csv("./data/test_simulate_interface.csv")

# %%
