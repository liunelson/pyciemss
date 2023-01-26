from typing import Dict, Optional

import torch
import pyro
import pyro.distributions as dist

from pyro.nn import pyro_method

from pyciemss.ODE.abstract import ODE, Time, State, Solution, Observation
from pyciemss.utils import state_flux_constraint

class SVIIvR(ODE):
    def __init__(self,
                N,
                noise_prior=dist.Uniform(5., 10.),
                beta_prior=dist.Uniform(0.1, 0.3),
                betaV_prior=dist.Uniform(0.025, 0.05),
                gamma_prior=dist.Uniform(0.05, 0.35),
                gammaV_prior=dist.Uniform(0.1, 0.4),
                nu_prior=dist.Uniform(0.001, 0.01)
                ):
        super().__init__()

        self.N = N
        self.noise_prior  = noise_prior
        self.beta_prior   = beta_prior
        self.betaV_prior  = betaV_prior
        self.gamma_prior  = gamma_prior
        self.gammaV_prior = gammaV_prior
        self.nu_prior     = nu_prior

    @pyro_method
    def deriv(self, t: Time, state: State) -> State:
        S, V, I, Iv, R = state

        # Local fluxes exposed to pyro for interventions.
        # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
        SV_flux_  = pyro.deterministic("SV_flux %f" % (t),  self.nu * S)
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  self.beta  * S * (I + Iv) / self.N)
        VIv_flux_ = pyro.deterministic("VIv_flux %f" % (t), self.betaV * V * (I + Iv) / self.N)
        IR_flux_  = pyro.deterministic("IR_flux %f" % (t),  self.gamma * I)
        IvR_flux_ = pyro.deterministic("IvR_flux %f" % (t), self.gammaV * Iv)

        # these state_flux_constraints ensure that we don't have vaccinated people become susceptible, etc.
        SV_flux = state_flux_constraint(S,  SV_flux_)
        SI_flux = state_flux_constraint(S,  SI_flux_)
        VIv_flux = state_flux_constraint(V,  VIv_flux_)
        IR_flux = state_flux_constraint(I, IR_flux_)
        IvR_flux = state_flux_constraint(Iv, IvR_flux_)

        # Where the real magic happens.
        dSdt  = -SI_flux - SV_flux
        dVdt  = -VIv_flux + SV_flux
        dIdt  = SI_flux - IR_flux
        dIvdt = VIv_flux - IvR_flux
        dRdt  = IR_flux + IvR_flux

        return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self) -> None:

        self.noise     = pyro.sample("noise", self.noise_prior)
        self.beta      = pyro.sample("beta", self.beta_prior)
        self.betaV     = pyro.sample("betaV", self.betaV_prior)
        self.gamma     = pyro.sample("gamma", self.gamma_prior)
        self.gammaV    = pyro.sample("gammaV", self.gammaV_prior)
        self.nu        = pyro.sample("nu", self.nu_prior)

    @pyro_method
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Solution:
        S, V, I, Iv, R = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise).to_event(1), obs=data["R_obs"])

        return (S_obs, V_obs, I_obs, R_obs)




class SVIIvR_simple(ODE):
    def __init__(self,
                N,
                noise_var_prior=dist.Uniform(5., 10.),
                betaSI_prior=dist.Uniform(0.1, 0.3),
                betaSIv_prior=dist.Uniform(0.1, 0.3),
                betaVI_prior=dist.Uniform(0.025, 0.05),
                betaVIv_prior=dist.Uniform(0.025, 0.05),
                gamma_prior=dist.Uniform(0.05, 0.35),
                gammaV_prior=dist.Uniform(0.1, 0.4),
                nu_prior=dist.Uniform(0.001, 0.01)
                ):
        super().__init__()

        self.N = N
        self.noise_var_prior  = noise_var_prior
        self.betaSI_prior   = betaSI_prior
        self.betaSIv_prior  = betaSIv_prior
        self.betaVI_prior   = betaVI_prior
        self.betaVIv_prior  = betaVIv_prior
        self.gamma_prior  = gamma_prior
        self.gammaV_prior = gammaV_prior
        self.nu_prior     = nu_prior

    @pyro_method
    def deriv(self, t: Time, state: State) -> State:
        S, V, I, Iv, R = state

        # Local fluxes exposed to pyro for interventions.
        # Note: This only works with solvers that use fixed time increments, such as Euler's method. Otherwise, we have name collisions.
        SV_flux_  = pyro.deterministic("SV_flux %f" % (t),  self.nu * S)
        SI_flux_  = pyro.deterministic("SI_flux %f" % (t),  self.betaSI  * S * I / self.N)
        SIv_flux_  = pyro.deterministic("SIv_flux %f" % (t),  self.betaSIv  * S * Iv / self.N)
        VI_flux_ = pyro.deterministic("VI_flux %f" % (t), self.betaVI * V * I / self.N)
        VIv_flux_ = pyro.deterministic("VIv_flux %f" % (t), self.betaVIv * V * Iv / self.N)
        IR_flux_  = pyro.deterministic("IR_flux %f" % (t),  self.gamma * I)
        IvR_flux_ = pyro.deterministic("IvR_flux %f" % (t), self.gammaV * Iv)

        # these state_flux_constraints ensure that we don't have vaccinated people become susceptible, etc.
        SV_flux  = state_flux_constraint(S,  SV_flux_)
        SI_flux  = state_flux_constraint(S,  SI_flux_)
        SIv_flux = state_flux_constraint(S,  SIv_flux_)
        VI_flux  = state_flux_constraint(V,  VI_flux_)
        VIv_flux = state_flux_constraint(V,  VIv_flux_)
        IR_flux  = state_flux_constraint(I,  IR_flux_)
        IvR_flux = state_flux_constraint(Iv, IvR_flux_)

        # Where the real magic happens.
        dSdt  = -SI_flux - SV_flux - SIv_flux
        dVdt  = -VIv_flux + SV_flux - VI_flux
        dIdt  = SI_flux - IR_flux + VI_flux
        dIvdt = VIv_flux - IvR_flux + SIv_flux
        dRdt  = IR_flux + IvR_flux

        return dSdt, dVdt, dIdt, dIvdt, dRdt

    @pyro_method
    def param_prior(self) -> None:

        self.noise_var = pyro.sample("noise_var", self.noise_var_prior)
        self.betaSI    = pyro.sample("betaSI", self.betaSI_prior)
        self.betaSIv   = pyro.sample("betaSIv", self.betaSIv_prior)
        self.betaVI    = pyro.sample("betaVI", self.betaVI_prior)
        self.betaVIv   = pyro.sample("betaVIv", self.betaVIv_prior)
        self.gamma     = pyro.sample("gamma", self.gamma_prior)
        self.gammaV    = pyro.sample("gammaV", self.gammaV_prior)
        self.nu        = pyro.sample("nu", self.nu_prior)

    @pyro_method
    def observation_model(self, solution: Solution, data: Optional[Dict[str, State]] = None) -> Solution:
        S, V, I, Iv, R = solution

        # It's a little clunky that we have to do `None` handling for each implementation of 'observation_model'...
        if data == None:
            data = {k: None for k in ["S_obs", "V_obs", "I_obs", "R_obs"]}

        # TODO: Make sure observations are strictly greater than 0.

        S_obs = pyro.sample("S_obs", dist.Normal(S, self.noise_var).to_event(1), obs=data["S_obs"])
        V_obs = pyro.sample("V_obs", dist.Normal(V, self.noise_var).to_event(1), obs=data["V_obs"])
        # We only observe the total number of infected people we don't know which of them are vaccinated.
        I_obs = pyro.sample("I_obs", dist.Normal(I + Iv, self.noise_var).to_event(1), obs=data["I_obs"])
        R_obs = pyro.sample("R_obs", dist.Normal(R, self.noise_var).to_event(1), obs=data["R_obs"])

        return (S_obs, V_obs, I_obs, R_obs)
