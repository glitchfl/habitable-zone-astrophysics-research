"""the 3x3 set of (A, eps) atmosphere scenarios - test more than just earth-like atmospheres"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from model import atmosphere_factor

ALBEDO_NAMES = ("lowA", "midA", "highA")
GREENHOUSE_NAMES = ("weakGH", "midGH", "strongGH")

# the scenario the report and figs 6/8 quote as the headline "big change" case
STRONG_GREENHOUSE = "midA_strongGH"


@dataclass
class Scenarios:
    albedo: np.ndarray
    epsilon: np.ndarray
    atmosphere_factor: np.ndarray   # sqrt((1-A)/(4 eps)) per scenario - the single number A and eps reach a(T) through
    names: list[str]
    reference_index: int

    @property
    def scenario_count(self) -> int:
        return len(self.albedo)

    # walk the 9 scenarios a row at a time instead of zipping the parallel arrays at every call site
    def __iter__(self):
        return zip(self.names, self.albedo, self.epsilon)


def build_scenarios(config: Config) -> Scenarios:
    """
    gets the config (which holds the 3 albedo levels and the 3 greenhouse levels)
    returns the 9 atmosphere scenarios (every albedo paired with every greenhouse)
    with the Earth-like reference moved to index 0

    every shift measured later is measured against that reference so pinning it to a
    known fixed slot removes a whole category of off-by-one bug
    """
    albedo_values, epsilon_values, scenario_names = [], [], []
    for albedo_name, albedo in zip(ALBEDO_NAMES, config.albedo_levels, strict=True):
        for greenhouse_name, epsilon in zip(GREENHOUSE_NAMES, config.epsilon_levels, strict=True):
            albedo_values.append(albedo)
            epsilon_values.append(epsilon)
            scenario_names.append(f"{albedo_name}_{greenhouse_name}")

    albedo_values = np.array(albedo_values)
    epsilon_values = np.array(epsilon_values)

    # compare with a tolerance (never use == on floats)
    is_reference = (np.abs(albedo_values - config.albedo_ref) < 1e-12) & (np.abs(epsilon_values - config.epsilon_ref) < 1e-12)
    if not is_reference.any():
        raise ValueError(
            f"reference (A={config.albedo_ref}, eps={config.epsilon_ref}) is not in the scenario grid - "
            "add it to albedo_levels/epsilon_levels or the deltas would be measured against nothing"
        )

    # the reference row first, then the other eight in their original order
    reference_first = np.concatenate([np.flatnonzero(is_reference)[:1], np.flatnonzero(~is_reference)])
    scenario_names = [scenario_names[index] for index in reference_first]
    scenario_names[0] += "_REF"

    albedo_values = albedo_values[reference_first]
    epsilon_values = epsilon_values[reference_first]

    return Scenarios(albedo=albedo_values, epsilon=epsilon_values, atmosphere_factor=atmosphere_factor(albedo_values, epsilon_values), names=scenario_names, reference_index=0)
