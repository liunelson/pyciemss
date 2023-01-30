import os

import torch
import pyro
import pytest

import numpy as np
import pyro.distributions as dist
from pyro.poutine import trace, replay, block
from pyro.infer.autoguide.guides import AutoDelta, AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

from pyciemss.ODE.base import PetriNetODESystem


MIRA_PATH = os.environ.get("MIRA_PATH", "test/models/evaluation_examples/")


@pytest.mark.parametrize("filename", [
    "CHIME-SVIIvR/model_petri.json",
    "CHIME-SIR/model_petri.json",
    "Bucky/model_petri.json",
])
def test_load_starterkit_scenarios_from_json(filename):
    filename = os.path.join("test/models/starter_kit_examples", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0


@pytest.mark.parametrize("filename", [
    "scenario1_all_ages.json",
    "scenario1_sir.json",
    "scenario1_three_ages.json",
])
def test_load_evaluation_scenario1_from_json(filename):
    filename = os.path.join(MIRA_PATH, "scenario_1/ta_2", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    if hasattr(model, "default_initial_state"):
        tspan = torch.linspace(0, 100, 100)
        solution, _ = model(model.default_initial_state, tspan)
        assert len(solution) == len(model.default_initial_state)
        assert solution[0].shape[0] == tspan.shape[0]


@pytest.mark.parametrize("filename", [
    "scenario2_sidarthe.json",
    "scenario2_sidarthe_v.json",
])
def test_load_evaluation_scenario2_from_json(filename):
    filename = os.path.join(MIRA_PATH, "scenario_2/ta_2", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    assert hasattr(model, "default_initial_state")
    tspan = torch.linspace(0, 100, 100)
    solution, _ = model(model.default_initial_state, tspan)
    assert len(solution) == len(model.default_initial_state)
    assert solution[0].shape[0] == tspan.shape[0]


@pytest.mark.parametrize("filename", [
    "scenario3_biomd958.json",
    "scenario3_biomd960.json",
])
def test_load_evaluation_scenario3_from_json(filename):
    filename = os.path.join(MIRA_PATH, "scenario_3/ta_2", filename)
    model = PetriNetODESystem.from_mira(filename)
    assert len(model.var_order) > 0
    if hasattr(model, "default_initial_state"):
        tspan = torch.linspace(0, 100, 100)
        solution, _ = model(model.default_initial_state, tspan)
        assert len(solution) == len(model.default_initial_state)
        assert solution[0].shape[0] == tspan.shape[0]
