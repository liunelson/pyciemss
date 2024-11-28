# %%[markdown]
# # Test Optimize Interface
#
# Run Optimize and do exploratory analysis of the results.


# %%
import os
import pyciemss
import torch
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy

import pyciemss.visuals.plots as plots
import pyciemss.visuals.vega as vega
import pyciemss.visuals.trajectories as trajectories
from pyciemss.integration_utils.observation import load_data
from pyciemss.integration_utils.intervention_builder import (
    param_value_objective,
    start_time_objective,
)
from pyciemss.ouu.qoi import obs_nday_average_qoi, obs_max_qoi

# import mira
from mira.modeling.viz import GraphicalModel
from mira.modeling.amr.petrinet import template_model_to_petrinet_json
from mira.metamodel import *

# %%
# Define a SIR model

parameters = {k: Parameter(name = k, value = 1.0) for k in ('b', 'g', 'S0', 'I0', 'R0')}
concepts = {k: Concept(name = k) for k in ('S', 'I', 'R')}
b, g, S, I, R = sympy.symbols('b g S I R')
initials = {k: Initial(concept = c, expression = safe_parse_expr(f'{k}0')) for k, c in concepts.items()}

# Default configuration
parameters['b'].value = 0.20
parameters['g'].value = 0.06
parameters['S0'].value = 0.9
parameters['I0'].value = 0.1
parameters['R0'].value = 0.0

parameters['b'].distribution = Distribution(type = 'StandardUniform1', parameters = {'minimum': 0.18, 'maximum': 0.22})
parameters['g'].distribution = Distribution(type = 'StandardUniform1', parameters = {'minimum': 0.055, 'maximum': 0.065})


model = TemplateModel(
    templates = [
        ControlledConversion(
            subject = concepts['S'],
            outcome = concepts['I'],
            controller = concepts['I'],
            rate_law = b * S * I
        ),
        NaturalConversion(
            subject = concepts['I'],
            outcome = concepts['R'],
            rate_law = g * I
        )
    ],
    parameters = parameters,
    initials = initials,
    # observables = {'R0': Observable(name = 'R0', description = 'Basic reproduction number', expression = safe_parse_expr('b / g'))}
)

GraphicalModel.for_jupyter(model)

# %%
# Common simulation settings

start_time = 0.0
end_time = 40.0
logging_step_size = 1.0
num_samples = 100

# %%
# Define initial static interventions
static_parameter_interventions = {}

# static_parameter_interventions = {
#     0.1: {
#         'b': torch.tensor(0.6),
#         'g': torch.tensor(0.5),
#         'S0': torch.tensor(0.99),
#         'I0': torch.tensor(0.01),
#         'R0': torch.tensor(0.0),
#     }
# }

# %%
# Sample before optimize

result_preoptimize = pyciemss.sample(
    template_model_to_petrinet_json(model),
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    static_parameter_interventions = static_parameter_interventions,
    solver_method = "dopri5"
)

# %%
data = result_preoptimize['data']
data = data[data['sample_id'] == 0]
fig, ax = plt.subplots(2, 1, figsize = (8, 8))
x = data['timepoint_unknown']
for k in ('b', 'g', 'S0', 'I0', 'R0'):
    __ = ax[0].plot(x, data[f'persistent_{k}_param'], label = k)
for k in ('S', 'I', 'R'):
    __ = ax[1].plot(x, data[f'{k}_state'], label = k)

__ = ax[0].legend()
__ = ax[1].legend()

# %%
# Optimize settings

num_samples_ouu = 1000
maxiter = 1
maxfeval = 15

# %%
# Define interventions

# Intervention policy
intervention_time = [torch.tensor(5.0)]
intervened_params = ['b']

initial_guess_interventions = [0.2]
bounds_interventions = [[0.01], [1.0]]
static_parameter_interventions = param_value_objective(
    param_name = intervened_params,
    start_time = intervention_time,
)

# Objective function
param_current = [0.2]
objfun = lambda x: np.sum(np.abs(x - param_current[0]))

# Success criteria
observed_params = [['I_state']]
risk_bound = [0.22]
alpha = [0.95]
qoi = [lambda y: obs_nday_average_qoi(y, observed_params[0], 1)]


# %%
result_optimize = pyciemss.optimize(
    template_model_to_petrinet_json(model),
    end_time,
    logging_step_size,
    qoi,
    risk_bound,
    static_parameter_interventions,
    objfun,
    initial_guess_interventions = initial_guess_interventions,
    bounds_interventions = bounds_interventions,
    start_time = start_time,
    n_samples_ouu = num_samples_ouu,
    maxiter = maxiter,
    maxfeval = maxfeval,
    solver_method = 'dopri5',
)

print(result_optimize)

# %%
optimal_interventions = {
    intervention_time[0]: {'b': result_optimize['policy'][0]}
}

print("Optimal intervention: ", optimal_interventions)

# %%
# Run Sample after Optimize

result_postoptimize = pyciemss.sample(
    template_model_to_petrinet_json(model),
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    static_parameter_interventions = optimal_interventions,
    solver_method = 'dopri5'
)

# %%
# Repeat for QOI = Max

qoi = [lambda y: obs_max_qoi(y, observed_params[0])]

result_optimize_max = pyciemss.optimize(
    template_model_to_petrinet_json(model),
    end_time,
    logging_step_size,
    qoi,
    risk_bound,
    static_parameter_interventions,
    objfun,
    initial_guess_interventions = initial_guess_interventions,
    bounds_interventions = bounds_interventions,
    start_time = start_time,
    n_samples_ouu = num_samples_ouu,
    maxiter = maxiter,
    maxfeval = maxfeval,
    solver_method = "dopri5",
)

optimal_interventions_max = {
    intervention_time[0]: {'b': result_optimize_max['policy'][0]}
}

result_postoptimize_max = pyciemss.sample(
    template_model_to_petrinet_json(model),
    end_time,
    logging_step_size,
    num_samples,
    start_time = start_time,
    static_parameter_interventions = optimal_interventions_max,
    solver_method = 'dopri5'
)

# %%
colors = np.array([mpl.colormaps.get_cmap('tab10')(i) for i in np.linspace(0, 1, 10 + 1)])

fig, axes = plt.subplots(2, 1, figsize = (8, 8))

# Risk bound
__ = axes[1].plot([start_time, end_time], [risk_bound, risk_bound], linestyle = ':', color = 'k', label = 'Risk Bound')

# Before optimization
for i in range(num_samples):
    # data = result_preoptimize["data"].groupby(["timepoint_id"]).mean()
    data = result_preoptimize['data'][result_preoptimize['data']['sample_id'] == i]

    x = data["timepoint_unknown"]
    y = data["persistent_b_param"]
    __ = axes[0].plot(x, y, linestyle = "-", color = colors[0, :], label = "Pre-Opt", lw = 0.2, alpha = 0.2)

    x = data["timepoint_unknown"]
    y = data["I_state"]
    __ = axes[1].plot(x, y, linestyle = "-", color = colors[0, :], label = "Pre-Opt", lw = 0.2, alpha = 0.2)

    # After optimization
    # data = result_postoptimize["data"].groupby(["timepoint_id"]).mean()
    data = result_postoptimize['data'][result_postoptimize['data']['sample_id'] == i]

    x = data["timepoint_unknown"]
    y = data["persistent_b_param"]
    __ = axes[0].plot(x, y, linestyle = "-", color = colors[1, :], label = "QOI = Last N", lw = 0.2, alpha = 0.2)

    x = data["timepoint_unknown"]
    y = data["I_state"]
    __ = axes[1].plot(x, y, linestyle = "-", color = colors[1, :], label = "QOI = Last N", lw = 0.2, alpha = 0.2)

    # QOI = MAX
    # data = result_postoptimize_max["data"].groupby(["timepoint_id"]).mean()
    data = result_postoptimize_max['data'][result_postoptimize_max['data']['sample_id'] == i]

    x = data["timepoint_unknown"]
    y = data["persistent_b_param"]
    __ = axes[0].plot(x, y, linestyle = "-", color = colors[2, :], label = "QOI = Max", lw = 0.2, alpha = 0.2)

    x = data["timepoint_unknown"]
    y = data["I_state"]
    __ = axes[1].plot(x, y, linestyle = "-", color = colors[2, :], label = "QOI = Max", lw = 0.2, alpha = 0.2)


axes[0].legend(handles = 
    [
        mpl.lines.Line2D([0], [0], color = colors[0, :], lw = 1, label = 'Pre-Opt'),
        mpl.lines.Line2D([0], [0], color = colors[1, :], lw = 1, label = 'Post-Opt, QoI = Last N'),
        mpl.lines.Line2D([0], [0], color = colors[2, :], lw = 1, label = 'Post-Opt, QoI = Max')

    ], loc = 'upper right')


# %%
# Plot superquantiles
fig, axes = plt.subplots(2, 1, figsize = (8, 8))

for i, ax in enumerate(axes):

    # Last N
    if i == 0:
        # Pre-Opt
        # np.mean(dataQoI[:, -ndays:], axis=1)
        t = result_preoptimize['data']['timepoint_id'].iloc[-1]
        data = result_preoptimize['data'][result_preoptimize['data']['timepoint_id'] == t]['I_state']

        # Post-Opt
        t = result_postoptimize['data']['timepoint_id'].iloc[-1]
        data_ = result_postoptimize['data'][result_preoptimize['data']['timepoint_id'] == t]['I_state']


    if i == 1:
        # Pre-Opt
        data = result_preoptimize['data'].groupby(['sample_id']).max()['I_state']

        # Post-Opt
        data_ = result_postoptimize_max['data'].groupby(['sample_id']).max()['I_state']


    m1 = min(data.min(), data_.min())
    m2 = max(data.max(), data_.max())
    
    # Pre-Opt
    h, b = np.histogram(data, range = (m1, m2))
    x = 0.5 * (b[1:] + b[:-1])
    __ = ax.barh(x, h, height = 0.90 * (x[1] - x[0]), label = 'Pre-Opt', alpha = 0.5, color = 'grey')
    __ = ax.legend(loc = 'upper right')


    # Post-Opt
    h, b = np.histogram(data_, range = (m1, m2))
    x = 0.5 * (b[1:] + b[:-1])
    __ = ax.barh(x, h, height = 0.90 * (x[1] - x[0]), label = 'Post-Opt', alpha = 0.5, color = colors[2, :])
    # __ = plt.setp(ax, xlim = (b[0], b[-1]))

    # Risk bound
    __ = ax.plot([0, max(h) * 1.1], [risk_bound, risk_bound], color = 'k', linestyle = '--', label = 'Threshold')
    __ = ax.legend(loc = 'upper right')
    __ = plt.setp(ax, xlim = (0, max(h) * 1.1), ylim = (0, 0.5))

__ = plt.setp(axes[0], title = 'QoI = Last N')
__ = plt.setp(axes[1], title = 'QoI = Max')

# %%
