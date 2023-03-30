import unittest

import os
import torch
from copy import deepcopy

from pyciemss.PetriNetODE.base import MiraPetriNetODESystem, BetaNoisePetriNetODESystem
from pyciemss.PetriNetODE.events import ObservationEvent, LoggingEvent, StartEvent, StaticParameterInterventionEvent

from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import pyro

class TestODE(unittest.TestCase):
    '''Tests for the ODE module.'''

    # Setup for the tests
    def setUp(self):
        STARTERKIT_PATH = "test/models/starter_kit_examples/"
        filename = "CHIME-SIR/model_petri.json"
        filename = os.path.join(STARTERKIT_PATH, filename)
        self.model = BetaNoisePetriNetODESystem.from_mira(filename)

    # Clean up after tests
    def tearDown(self):
        self.model = None

    def test_from_mira(self):
        '''Test the from_mira method.'''
        STARTERKIT_PATH = "test/models/starter_kit_examples/"
        filename = "CHIME-SIR/model_petri.json"
        filename = os.path.join(STARTERKIT_PATH, filename)
        model = MiraPetriNetODESystem.from_mira(filename)
        self.assertIsNotNone(model)

    def test_from_mira_with_noise(self):
        self.assertIsNotNone(self.model)

    def test_load_remove_start_event(self):
        '''Test the load_event method for StartEvent and the remove_start_event methods.'''
        event = StartEvent(0.0, {"S": 0.9, "I": 0.1, "R": 0.0})

        self.model.load_event(event)
        
        self.assertEqual(len(self.model._static_events), 1)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(0.0))
        self.assertEqual(self.model._static_events[0].initial_state["S"], torch.tensor(0.9))
        self.assertEqual(self.model._static_events[0].initial_state["I"], torch.tensor(0.1))
        self.assertEqual(self.model._static_events[0].initial_state["R"], torch.tensor(0.0))
        
        self.model.remove_start_event()
        
        self.assertEqual(len(self.model._static_events), 0)

    def test_load_remove_logging_event(self):
        '''Test the load_events method for LoggingEvent and the remove_logging_events methods.'''
        self.model.load_events([LoggingEvent(1.0), LoggingEvent(2.0)])

        self.assertEqual(len(self.model._static_events), 2)
        self.assertEqual(self.model._static_events[0].time, 1.0)
        self.assertEqual(self.model._static_events[1].time, 2.0)

        self.model.remove_logging_events()

        self.assertEqual(len(self.model._static_events), 0)

    def test_load_remove_observation_events(self):
        '''Test the load_observation_events and the remove_observation_events methods.'''
        observation1 = ObservationEvent(0.01, {"S": 0.9, "I": 0.1})
        observation2 = ObservationEvent(1.0, {"S": 0.8})

        self.model.load_events([observation1, observation2])
        
        self.assertEqual(len(self.model._static_events), 2)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(0.01))
        self.assertEqual(self.model._static_events[1].time, torch.tensor(1.0))
        self.assertEqual(self.model._static_events[0].observation["S"], torch.tensor(0.9))
        self.assertEqual(self.model._static_events[0].observation["I"], torch.tensor(0.1))
        self.assertEqual(self.model._static_events[1].observation["S"], torch.tensor(0.8))

        self.assertEqual(set(self.model._observation_var_names), {"S", "I"})

        self.model.remove_observation_events()

        self.assertEqual(len(self.model._static_events), 0)

        self.assertEqual(self.model._observation_var_names, [])

    def test_load_remove_static_parameter_intervention_events(self):
        '''Test the load_events method for StaticParameterIntervention and the remove_static_parameter_interventions methods.'''
        # Load some static parameter intervention events
        intervention1 = StaticParameterInterventionEvent(2.99, "beta", 0.0)
        intervention2 = StaticParameterInterventionEvent(4.11, "beta", 10.0)
        self.model.load_events([intervention1, intervention2])

        self.assertEqual(len(self.model._static_events), 2)

        self.assertEqual(self.model._static_events[0].time, torch.tensor(2.99))
        self.assertEqual(self.model._static_events[1].time, torch.tensor(4.11))
        self.assertEqual(self.model._static_events[0].parameter, "beta")
        self.assertEqual(self.model._static_events[1].parameter, "beta")
        self.assertEqual(self.model._static_events[0].value, torch.tensor(0.0))
        self.assertEqual(self.model._static_events[1].value, torch.tensor(10.0))

        self.model.remove_static_parameter_intervention_events()
        
        self.assertEqual(len(self.model._static_events), 0)

    def test_observation_indices_and_values(self):
        '''Test the _setup_observation_indices_and_values method.'''

        observation1 = ObservationEvent(0.01, {"S": 0.9, "I": 0.1})
        observation2 = ObservationEvent(1.0, {"S": 0.8})

        self.model.load_events([observation1, observation2])

        self.assertListEqual(self.model._observation_var_names, ["S", "I"])

        self.model._setup_observation_indices_and_values()

        self.assertEqual(self.model._observation_indices["S"], [0, 1])
        self.assertEqual(self.model._observation_indices["I"], [0])

        self.assertTrue(torch.equal(self.model._observation_values["S"], torch.tensor([0.9, 0.8]))) 
        self.assertTrue(torch.equal(self.model._observation_values["I"], torch.tensor([0.1])))

    def test_integration(self):

        model = self.model

        # Load the start event
        start_event = StartEvent(0.0, {"S": 0.9, "I": 0.1, "R": 0.0})
        model.load_event(start_event)

        # Load the logging events
        tspan = range(1, 10)
        logging_events = [LoggingEvent(t) for t in tspan]
        model.load_events(logging_events)

        # Run the model without observations
        solution = model()

        self.assertEqual(len(solution["I"]), len(solution["R"]))
        self.assertEqual(len(solution["I"]), len(solution["S"]))
        self.assertEqual(len(solution["I"]), len(tspan))

        # Susceptible individuals should decrease over time
        self.assertTrue(torch.all(solution["S"][:-1] > solution["S"][1:]))

        # Recovered individuals should increase over time
        self.assertTrue(torch.all(solution["R"][:-1] < solution["R"][1:]))

        # Remove the logs
        model.remove_logging_events()

        # Load the observation events
        observation1 = ObservationEvent(0.01, {"S": 0.9, "I": 0.1})
        observation2 = ObservationEvent(1.0, {"S": 0.8})

        self.model.load_events([observation1, observation2])

        solution = model()

        # No logging events, so we don't return anything.
        self.assertEqual(len(solution["I"]), len(solution["R"]))
        self.assertEqual(len(solution["I"]), len(solution["S"]))
        self.assertEqual(len(solution["I"]), 0)

        # Run inference
        guide = AutoNormal(model)

        optim = Adam({'lr': 0.03})
        loss_f = Trace_ELBO(num_particles=1)

        svi = SVI(model, guide, optim, loss=loss_f)

        pyro.clear_param_store()

        # Step once to setup parameters
        svi.step()
        old_params = deepcopy(list(guide.parameters()))

        # Check that the parameters have been set
        self.assertEqual(len(old_params), 4)

        # Step again to update parameters
        svi.step()

        # Check that the parameters have been updated
        for (i, p) in enumerate(guide.parameters()):
            self.assertNotEqual(p, old_params[i])
        
        # Remove the observation events and add logging events.
        model.remove_observation_events()
        model.load_events(logging_events)

        # Add a few static parameter interventions
        intervention1 = StaticParameterInterventionEvent(2.99, "beta", 0.0)
        intervention2 = StaticParameterInterventionEvent(4.11, "beta", 10.0)
        model.load_events([intervention1, intervention2])

        # Sample from the posterior predictive distribution  
        predictions = Predictive(model, guide=guide, num_samples=2)()

        self.assertEqual(predictions['I_sol'].shape, predictions['R_sol'].shape)
        self.assertEqual(predictions['I_sol'].shape, predictions['S_sol'].shape)
        self.assertEqual(predictions['I_sol'].shape, torch.Size([2, 9]))

        # Susceptible individuals shouldn't change between t=3 and t=4 because of the first intervention
        self.assertTrue(torch.all(predictions['S_sol'][:, 2] == predictions['S_sol'][:, 3]))

        # Recovered individuals should increase between t=3 and t=4
        self.assertTrue(torch.all(predictions['R_sol'][:, 2] < predictions['R_sol'][:, 3]))

        # Susceptible individuals should decrease between t=4 and t=5 because of the second intervention
        self.assertTrue(torch.all(predictions['S_sol'][:, 3] > predictions['S_sol'][:, 4]))