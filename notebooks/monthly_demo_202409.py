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


# %%[markdown]
# ## Problem 1
# 
# Starting model: SIRHD
# Configuration: 
# * December 1, 2021 to March 1, 2022
# * Training period: December 2021


# %%
# Define SIRHD model from given equations
with open("./data/monthly_demo_202409/SIRHD.json", "r") as f:
    model_sirhd = template_model_from_amr_json(json.load(f))

# %%
# Load datasets
dtype = {
    "date": str,
    "location": str, 
    "location_name": str,
    "value": int
}
datasets = {
    "inccases": pandas.read_csv(os.path.join(DATAPATH, "truth-Incident Cases.csv"), dtype = dtype),
    "cumdeaths": pandas.read_csv(os.path.join(DATAPATH, "truth-Cumulative Deaths.csv"), dtype = dtype),
    "inchosps": pandas.read_csv(os.path.join(DATAPATH, "truth-Incident Hospitalizations.csv"), dtype = dtype),
}

# %%
# Convert to prevalence
prevalence = {}
for k, df in datasets.items():

    # Just national data
    df_ = df[df["location_name"] == "United States"].sort_values("date").reset_index()
    df_ = df_.set_index("date")

    match k:
        case "inccases":

            # Recovery time = 1 / 0.07 ~ 14 days
            # Sum incident cases, 14 days before current date
            x = df_["value"].rolling(14).sum().dropna()
            prevalence["I"] = x.values
            prevalence["timeI"] = x.index.values

            # Recovered = cum sum of inc cases (14 days before today) - current deaths
            x = df_["value"].cumsum().iloc[:(-(14 - 1))].dropna()
            prevalence["R"] = x.values
            prevalence["timeR"] = df_[(14 - 1):].index.values

        case "inchosps":

            # Hospitalized recovery time = 1 / 0.07 ~ 14 days
            # Hospitalized death time = 1 / 0.3 ~ 3 days
            # Average time to exit hospitalized state ~ 1 / (0.87 * 0.07 + 0.13 * 0.3) ~ 10 days
            x = df_["value"].rolling(10).sum().dropna()
            prevalence["H"] = x.values
            prevalence["timeH"] = x.index.values

        case "cumdeaths":
            prevalence["D"] = df_["value"].values
            prevalence["timeD"] = df_.index.values


prevalence

# %%
# Join time-series together with shared datetime
df = pandas.DataFrame()
for k in "SIRHD":
    if k in prevalence.keys():
        df_ = pandas.DataFrame({k: prevalence[k], "time": prevalence[f"time{k}"]}).set_index("time")
        if len(df) < 1:
            df = df_
        else:
            df = df.join(df_, on = "time", how = "inner").sort_index()

# Recovered = cum sum of inc cases (14 days before today) - current deaths
df["R"] = df["R"] - df["D"]

# Total pop = 150e6
# susceptible = total pop - I - R - H - D
df["N"] = 150e6
df["S"] = df["N"] - df["I"] - df["R"] - df["H"] - df["D"]

df = df.reset_index()
df

# %%
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

for k in "SIRHD":
    if k in df.columns:
        t = numpy.array(pandas.to_datetime(df['time']).dt.to_pydatetime())
        # t = numpy.array(pandas.to_datetime(df.index.to_series()).dt.to_pydatetime())
        __ = ax.plot(t, df[k], label = k)

__ = ax.legend()
__ = plt.setp(ax, yscale = 'log', title = 'Prevalence derived from ForecastHub Gold')
ax.tick_params(axis = 'x', labelrotation = 45)

# %%
df.to_csv(os.path.join(DATAPATH, "truthSIRHD.csv"))

# %%[markdown]
# ## Problem 5
# 
# Model A = SIR model from Hewitt 2024, stratified by county
# Model B = 
# Model C = 

# %%
# Model A
# 
# Configuration dataset from SI of Hewitt 2024



# %%
# Model C

model_c = {}
with open('./data/monthly_demo_202409/problem_5/modelC/Prob 5 Model C (new params, new rate laws, stratified, no H, replaced omega_W_W) (1).json', 'r') as fp:
    model_c['amr'] = json.load(fp)
    # model_c['tm'] = template_model_from_amr_json(model_c['amr'])


# %%
# Configuration from Agustin
with open('./data/monthly_demo_202409/problem_5/modelC/Captive deer (outdoor ranch), only humans initially infected/cae2fdb4-7671-4918-b374-c197ef07d04a.json', 'r') as fp:
    config = json.load(fp)

# %%
model = copy.deepcopy(model_c['amr'])

# %%
# Update the parameter values
m = {p['id']: i for i, p in enumerate(model['semantics']['ode']['parameters'])}
for p in config['parameterSemanticList']:
    i = m[p['referenceId']]
    model['semantics']['ode']['parameters'][i]['value'] = p['distribution']['parameters']['value']


# Update the initial values
m = {p['target']: i for i, p in enumerate(model['semantics']['ode']['initials'])}
for p in config['initialSemanticList']:
    i = m[p['target']]
    model['semantics']['ode']['initials'][i]['expression'] = p['expression']
    model['semantics']['ode']['initials'][i]['expression_mathml'] = p['expressionMathml']

# %%
tm = template_model_from_amr_json(copy.deepcopy(model))
x, y = generate_init_param_tables(tm)
x

# %%
y

# %%
with open('./data/monthly_demo_202409/problem_5/modelC/model.json', 'w') as fp:
    json.dump(model, fp, indent = 4)

# %%
# Simulate
start_time = 0.0
end_time = 100.0
logging_step_size = 1.0
num_samples = 10

results = pyciemss.sample(
    './data/monthly_demo_202409/problem_5/modelC/model.json', 
    end_time, 
    logging_step_size, 
    num_samples, 
    start_time = start_time
)

# Plot results
plot_simulate_results(results)

# %%
