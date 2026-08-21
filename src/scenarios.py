"""the 3x3 set of (A, eps) atmosphere scenarios - test more than just earth-like atmospheres"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config

ALBEDO_NAMES = ("lowA", "midA", "highA")
GREENHOUSE_NAMES = ("weakGH", "midGH", "strongGH")


@dataclass
class Scenarios:
    albedo: np.ndarray
    epsilon: np.ndarray
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
    for albedo_index, albedo in enumerate(config.A_levels):
        for greenhouse_index, epsilon in enumerate(config.eps_levels):
            albedo_values.append(albedo)
            epsilon_values.append(epsilon)
            scenario_names.append(f"{ALBEDO_NAMES[albedo_index]}_{GREENHOUSE_NAMES[greenhouse_index]}")

    albedo_values = np.array(albedo_values)
    epsilon_values = np.array(epsilon_values)

    # compare with a tolerance (never use == on floats)
    is_reference = (np.abs(albedo_values - config.A_ref) < 1e-12) & (np.abs(epsilon_values - config.eps_ref) < 1e-12)
    if not is_reference.any():
        raise ValueError(
            f"reference (A={config.A_ref}, eps={config.eps_ref}) is not in the scenario grid - "
            "add it to A_levels/eps_levels or the deltas would be measured against nothing"
        )

    reference_first = np.concatenate([np.flatnonzero(is_reference)[:1], np.flatnonzero(~is_reference)])
    scenario_names = [scenario_names[index] for index in reference_first]
    scenario_names[0] += "_REF"

    return Scenarios(albedo=albedo_values[reference_first], epsilon=epsilon_values[reference_first], names=scenario_names, reference_index=0)
