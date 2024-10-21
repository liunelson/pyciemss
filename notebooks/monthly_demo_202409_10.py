# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# # Monthly Demo (2024-09)
# 
# 1. 18-month evaluation, scenario 2, Q4-6, 8-12
# 2. 12-month hackathon, scenario 1
# 3. 12-month hacakthon, scenario 1, Q1a-c; scenario 2, Q2a,c
# 4. 12-month evaluation, scenario 3, Q1a-d
# 5. 18-month evaluation, scenario 1, Q1-3

# %%
import os
import json
import mira.metamodel
import numpy
import sympy
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import pytest
import torch
import copy

from tabulate import tabulate

import pyciemss
from pyciemss.integration_utils.intervention_builder import (
    combine_static_parameter_interventions,
    param_value_objective,
    start_time_objective,
    start_time_param_value_objective,
    intervention_func_combinator,
)
from pyciemss.ouu.qoi import obs_max_qoi

import mira
from mira.sources import biomodels
from mira.modeling.viz import GraphicalModel
from mira.modeling.amr.petrinet import template_model_to_petrinet_json
from mira.sources.amr.petrinet import template_model_from_amr_json
from mira.dkg.web_client import is_ontological_child_web
from mira.metamodel import *
# from mira.modeling import Model

from mira.modeling.amr.ops import *
from mira.metamodel.io import expression_to_mathml

MIRA_REST_URL = "http://34.230.33.149:8771/api"

# %%
DATAPATH = "./data/monthly_demo_202409"

# %%
# Helper functions

# Generate Sympy equations from a template model
def generate_odesys(model, latex: bool = False, latex_align: bool = False) -> list:

    odeterms = {var: 0 for var in model.get_concepts_name_map().keys()}

    for t in model.templates:
        if hasattr(t, "subject"):
            var = t.subject.name
            odeterms[var] -= t.rate_law.args[0]
        
        if hasattr(t, "outcome"):
            var = t.outcome.name
            odeterms[var] += t.rate_law.args[0]

    # Time
    symb = lambda x: sympy.Symbol(x)
    try:
        time = model.time.name
    except:
        time = "t"
    finally:
        t = symb(time)

    # Construct Sympy equations
    odesys = [
        sympy.Eq(sympy.diff(sympy.Function(var)(t), t), terms) 
        if latex == False
        else sympy.latex(sympy.Eq(sympy.diff(sympy.Function(var)(t), t), terms))
        for var, terms in odeterms.items()
    ]
    
    if (latex == True) & (latex_align == True):
        odesys = "\\begin{align*} \n    " + " \\\\ \n    ".join([eq.replace(" = ", " &= ") for eq in odesys]) + "\n\\end{align*}"
        # odesys = "\\begin{align*}     " + " \\\\    ".join([eq.replace(" = ", " &= ") for eq in odesys]) + "\\end{align*}"

    return odesys

# Generate summary table of a template model
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

# Generate initial condition and parameter tables
def generate_init_param_tables(model) -> tuple[pandas.DataFrame, pandas.DataFrame]:

    data = {}
    data["name"] = [name for name, __ in model.initials.items()]
    data["expression"] = [init.expression for __, init in model.initials.items()]
    df_initials = pandas.DataFrame(data)

    data = {}
    data["name"] = [name for name, __ in model.parameters.items()]
    data["value"] = [param.value for __, param in model.parameters.items()]
    df_params = pandas.DataFrame(data)

    return (df_initials, df_params)

# Plot pyciemss.simulate results
def plot_simulate_results(results: dict) -> None:

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

    return None

# %%
# ## Problem 5
# 
# Debug Model C variants created in Terarium

# %%
filepaths = [
    "./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws).json",
    "./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws, stratified).json",
    "./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws, stratified, no H).json",
    # "./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws, stratified, no H, replaced omega_W_W).json",
    "./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws, stratified, no H, replaced omega_W_W) 2.json"
]

models = []
for filepath in filepaths:
    with open(filepath, "r") as f:
        models.append(template_model_from_amr_json(json.load(f)))

# %%
# Simulate
start_time = 0.0
end_time = 10.0
logging_step_size = 1.0
num_samples = 1

for i, filepath in enumerate(filepaths):

    print(f"{i}: {filepath.split('/')[-1]}")

    try:
        results = pyciemss.sample(
            filepath, 
            end_time, 
            logging_step_size, 
            num_samples, 
            start_time = start_time
        )

        # Plot results
        # plot_simulate_results(results)

    except:
        pass

# %%
# Check the model configuration of model 3

i = 3
params, inits = generate_init_param_tables(models[i])

print(tabulate(params, headers = 'keys', tablefmt = 'psql'))
print(tabulate(inits, headers = 'keys', tablefmt = 'psql'))

# %%
models[i].get_all_used_parameters()

# Need to replace omega_W_W also in t2_W_W

# %%