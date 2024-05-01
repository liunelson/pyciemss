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
from typing import Dict, List, NoReturn, Optional
import pyciemss
import mira
import dill
import json
import pandas as pd
# from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl


from pyciemss.ouu.qoi import obs_nday_average_qoi
# def obs_nday_average_qoi(
#     samples: Dict[str, torch.Tensor], contexts: List, ndays: int = 7
# ) -> np.ndarray:
#     """
#     Return estimate of last n-day average of each sample.
#     samples is is the output from a Pyro Predictive object.
#     samples[VARIABLE] is expected to have dimension (nreplicates, ntimepoints)
#     Note: last ndays timepoints is assumed to represent last n-days of simulation.
#     """
#     dataQoI = samples[contexts[0]].detach().numpy()

#     return np.mean(dataQoI[:, -ndays:], axis=1)

# %%[markdown]
# ## 1. Optimize

# %%
# Specify model

MODELS_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/models/"
DATA_PATH = "https://raw.githubusercontent.com/DARPA-ASKEM/simulation-integration/main/data/datasets/"

model3 = os.path.join(MODELS_PATH, "SIR_stockflow.json")
dataset1 = os.path.join(DATA_PATH, "traditional.csv")

model3_tm = mira.sources.amr.model_from_url(model3)
model3_tm.draw_jupyter("model3.png")

# %%
# Specify time
start_time = 0.0
end_time = 50.0
logging_step_size = 0.5

# %%
# Define intervention

# intervention_time = torch.tensor(1.0)
intervention_time = torch.tensor(15.0)

intervened_params = "p_cbeta"
p_cbeta_current = model3_tm.parameters[intervened_params].value
# initial_guess_interventions = p_cbeta_current
initial_guess_interventions = 0.15
bounds_interventions = [[0.01], [1.0]]
# bounds_interventions = [
#     [model3_tm.parameters["p_cbeta"].distribution.parameters["minimum"]], 
#     [model3_tm.parameters["p_cbeta"].distribution.parameters["maximum"]]
# ]

# Define QoI
observed_params = ["I_state"]
qoi = lambda x: obs_nday_average_qoi(x, observed_params, 1)
risk_bound = 500.0

objfun = lambda x: np.abs(p_cbeta_current - x)
static_parameter_interventions = {intervention_time: intervened_params}

# %%
opt_result = pyciemss.optimize(
    model3, 
    end_time, 
    logging_step_size, 
    qoi, 
    risk_bound, 
    static_parameter_interventions, 
    objfun, 
    initial_guess_interventions = initial_guess_interventions, 
    bounds_interventions = bounds_interventions, 
    start_time = 0.0, 
    n_samples_ouu = int(1e2), 
    maxiter = 2, 
    maxfeval = 50, 
    solver_method = "euler"
)

# %%
print(f'Optimal policy for intervening on {static_parameter_interventions[list(static_parameter_interventions.keys())[0]]} is ', opt_result["policy"])

# %%
with open("opt_result.dill", "wb") as f:
    dill.dump(opt_result, f)

# %%
with open("opt_result.dill", "rb") as f:
    opt_result = dill.load(f)

# %%
# opt_result = 
#     policy = <torch.tensor> optimal policy parameter value
#     OptResults = <scipy.optimize._optimize.OptimizeResult>
#         message
#         success
#         fun
#         x
#         nit
#         minimization_failures
#         lowest_optimization_Result
#             success
#             status
#             function
#             x
#             nfev
#             maxcv


# ???
#  data: dataframe
#  policy
#    policy: optimal value of intervention parameter
#    OptResults: scipy optimize result message
#    risk: alpha-superquantile risk value (alpha = 0.95)
#    samples: sample values of each model parameter
#    qoi: num_samples of qoi values
#  quantiles: dataframe    

# %%
# Run simulation using results from Optimize

num_samples = 100

results_opt = pyciemss.sample(
    model3, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time, 
    static_parameter_interventions = {
        intervention_time: {intervened_params: opt_result["policy"]}
    }, 
    solver_method = "dopri5"
)

# %%
# Run baseline simulation
results_baseline = pyciemss.sample(
    model3, 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time, 
    solver_method = "dopri5"
)

# %%
# results = 
#     data = <pandas.DataFrame> simulation results in tabular form
#     unprocessed_result = <dict> simulation results in tensor form
#         persistent_p_cbeta = <torch.tensor>
#         persistent_p_tr = <torch.tensor>
#         R_state = <torch.tensor>
#         S_state = <torch.tensor>

# %%
# Plot
def plot_results(results: pd.DataFrame, agg: str = "mean", observed_param: Optional[str] = None, risk_bound: Optional[float] = None) -> NoReturn:

    cmap = mpl.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, 10))

    variables = [c for c in results["data"].columns if c.split("_")[-1] in ("sol", "state")]
    parameters = [c for c in results["data"].columns if c.split('_')[-1] == "param"]

    # parameters = ["persistent_p_cbeta_param", "persistent_p_cbeta_param"]

    fig, axes = plt.subplots(2, 1, figsize = (8, 8))

    # Trajectories
    for quantities, ax in zip((variables, parameters), axes):

        for k, v in enumerate(quantities):

            x = results["data"][["timepoint_id", "sample_id", v]].groupby(["timepoint_id"]).mean().index.values
            
            if agg == "mean":
                y = results["data"][["timepoint_id", "sample_id", v]].groupby(["timepoint_id"]).mean()[v].values
                yerr = results["data"][["timepoint_id", "sample_id", v]].groupby(["timepoint_id"]).sem()[v].values
                __ = ax.errorbar(x, y, yerr = yerr, label =  v.split("_")[0])

            if agg == "none":

                for s in results["data"]["sample_id"].unique():
                    y = results["data"][["timepoint_id", "sample_id", v]][results["data"]["sample_id"] == s][v]
                    
                    if s == 0:
                        __ = ax.plot(x, y, color = colors[k, :], alpha = 0.3, label = f'{"_".join(v.split("_")[:-1])}')
                    else:
                        __ = ax.plot(x, y, color = colors[k, :], alpha = 0.3)

        __ = plt.setp(ax, xlabel = 'Time', ylabel = 'quantities', xlim = [x[0], x[-1]])
        __ = ax.legend([mpl.lines.Line2D([0], [0], color = colors[k]) for k, v in enumerate(quantities)], [f'{"_".join(v.split("_")[:-1])}' for v in quantities], loc = "upper left")


    # Parameter distribution
    n = len(parameters)
    t = 2
    fig, axes = plt.subplots(n, n, figsize = (6, 6))
    fig.tight_layout()
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                p = parameters[i]
                __ = ax.hist(results["data"][results["data"]["timepoint_id"] == t][p], label = f'{"_".join(p.split("_")[:-1])}', color = 'k', alpha = 0.3)
                # __ = ax.legend(loc = "upper left")
                # __ = plt.setp(ax, ylim = (0, len(np.unique(results["data"]["sample_id"]))))
            
            else:
                px = parameters[j]
                py = parameters[i]

                x = results["data"][results["data"]["timepoint_id"] == t][px]
                y = results["data"][results["data"]["timepoint_id"] == t][py]

                __ = ax.scatter(x, y, marker = '.', color = 'k', alpha = 0.3)

            if i == (n - 1):
                __ = plt.setp(ax, xlabel = f'{"_".join(parameters[j].split("_")[:-1])}')
            
            if j == 0:
                __ = plt.setp(ax, ylabel = f'{"_".join(parameters[i].split("_")[:-1])}')

    # Distribution of QoI
    fig, ax = plt.subplots(1, 1, figsize = (4, 4))
    h, b = np.histogram(results["risk"][observed_param]["qoi"])
    ax.barh(b[:-1], h, height = b[1] - b[0], align = "center", label = "QoI Distribution")
    r = results_opt["risk"]["I_state"]["risk"]
    if risk_bound != None:
        ax.plot([0, max(h)], [risk_bound, risk_bound], label = "QoI Risk Bound", color = "k")
    if observed_params != None:
        ax.plot([0, max(h)], [r, r], label = "95% Risk Threshold")
    __ = plt.setp(ax, xlabel = "Sample Count")
    __ = ax.legend()

# %%
plot_results(results_opt, agg = "none", observed_param = "I_state", risk_bound = risk_bound)
plot_results(results_baseline, agg = "none", observed_param = "I_state", risk_bound = risk_bound)

# %%
plot_results(results_opt, agg = "mean")

# %%
print(f"States: {list(model3_tm.get_concepts_name_map().keys())}")
print(f"Parameters: {list(model3_tm.parameters.keys())}")

print(f"Transitions:")
print(f'\n'.join([' -> '.join([f'{getattr(t, role).name}' if role in t.concept_keys else f'{str(getattr(t, role))}' if role == "rate_law" else '*' for role in ["subject", "rate_law", "outcome"]]) for i, t in enumerate(model3_tm.templates)]))

# %%
h = "persistent_p_cbeta_param"
for s in results["data"]["sample_id"].unique():
    x = results["data"][["timepoint_id", "sample_id", h]][results["data"]["sample_id"] == s]["timepoint_id"]
    y = results["data"][["timepoint_id", "sample_id", h]][results["data"]["sample_id"] == s][h]
    plt.plot(x, y, color = 'k', alpha = 0.2)
