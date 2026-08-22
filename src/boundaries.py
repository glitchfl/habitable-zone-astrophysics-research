"""a_in and a_out for every star x scenario"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from data import Sample
from model import atmosphere_factor, hz_distance
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
