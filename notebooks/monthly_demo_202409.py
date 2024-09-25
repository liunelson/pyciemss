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

# %%[markdown]
# ## Problem 1
# 
# Starting model: SIRHD
# Configuration: 
# * December 1, 2021 to March 1, 2022
# * Training period: December 2021
# 

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


df

# %%
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

for k in "SIRHD":
    if k in df.columns:
        t = numpy.array(pandas.to_datetime(df.index.to_series()).dt.to_pydatetime())
        __ = ax.plot(t, df[k], label = k)

__ = ax.legend()
__ = plt.setp(ax, yscale = 'log')

# %%
df.to_csv(os.path.join(DATAPATH, "truthSIRHD.csv"))

# %%
