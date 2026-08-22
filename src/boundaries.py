"""a_in and a_out for every star x scenario (plus two independent numeric routes)"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from config import Config
from data import Sample
from model import atmosphere_factor, hz_distance, planet_temp
from scenarios import Scenarios


@dataclass
class Boundaries:
    a_in: np.ndarray                # (n_stars, n_scenarios), AU
    a_out: np.ndarray               # (n_stars, n_scenarios), AU
    atmosphere_factor: np.ndarray   # (n_scenarios,)

    @property
    def width(self) -> np.ndarray:
        return self.a_out - self.a_in


def compute_boundaries(sample: Sample, scenarios: Scenarios, config: Config) -> Boundaries:
    """
    gets the star sample the 9 atmosphere scenarios and the config
    returns a_in and a_out (in AU) for every star in every scenario (a table with one
    row per star and one column per scenario)

    a_in = a(T_hot) is the inner edge - too close and water boils
    a_out = a(T_cold) is the outer edge - too far and water freezes

    filled by broadcasting - the star arrays are shaped (n, 1) and the scenario arrays
    (1, m) so numpy expands them into the full (n, m) table with no python loop
    """
    teff_column = sample.teff[:, None]
    radius_column_m = sample.r_m[:, None]
    albedo_row = scenarios.albedo[None, :]
    epsilon_row = scenarios.epsilon[None, :]

    return Boundaries(
        a_in=hz_distance(teff_column, radius_column_m, config.T_hot, albedo_row, epsilon_row) / config.AU,
        a_out=hz_distance(teff_column, radius_column_m, config.T_cold, albedo_row, epsilon_row) / config.AU,
        atmosphere_factor=atmosphere_factor(scenarios.albedo, scenarios.epsilon),
    )


def hz_distance_brentq(teff: float, r_star_m: float, target_temp: float, albedo: float, epsilon: float, config: Config) -> float:
    """
    gets one star (T_eff, R_*) a wanted temperature the
    albedo A epsilon and the config
    returns the same distance hz_distance gives (in meters) but found by searching

    the proposal's second route - rather than rearranging the formula it hunts for the a
    where T_p(a) - target_temp crosses zero - the two agreeing to 1e-14 is the evidence that
    neither the algebra nor the code is wrong

    two deliberate choices:

    solved in log(a) because the search range spans twelve decades and linear space would
    be badly conditioned

    bracketed by fixed generic bounds rather than by the analytic answer - bracketing as
    [a_analytic/10, a_analytic*10] would feed the check the very number it is meant to be
    checking - T_p falls steadily with a so 1e-6 to 1e6 AU always contains the root
    whatever the star
    """
    def residual(log_distance: float) -> float:
        return planet_temp(teff, r_star_m, np.exp(log_distance), albedo, epsilon) - target_temp

    log_lower_bound = np.log(1e-6 * config.AU)
    log_upper_bound = np.log(1e6 * config.AU)
    return float(np.exp(brentq(residual, log_lower_bound, log_upper_bound, xtol=1e-15, rtol=8.9e-16, maxiter=200)))


def hz_distance_interp(teff: float, r_star_m: float, target_temp: float, albedo: float, epsilon: float, config: Config) -> float:
    """
    gets one star (T_eff in kelvin - R_* in meters) a wanted temperature (in kelvin) the
    albedo A epsilon and the config
    returns the same distance again (in meters) - this time by interpolation

    the proposal's third route - compute T_p at 40001 distances and read off where the
    curve crosses the target temperature

    interpolates in linear T on purpose - in log-log the relation is an exact straight
    line so the interpolation would be exact by construction and would test nothing
    """
    distance_grid_m = np.logspace(np.log10(1e-3 * config.AU), np.log10(1e3 * config.AU), 40001)
    temperature_grid = planet_temp(teff, r_star_m, distance_grid_m, albedo, epsilon)
    # Tp decreases with a, so reverse both for np.interp which needs increasing x
    return float(np.interp(target_temp, temperature_grid[::-1], distance_grid_m[::-1]))
