"""how far the boundaries move relative to the Earth-like reference scenario"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boundaries import Boundaries
from scenarios import Scenarios


@dataclass
class Sensitivity:
    inner_shift_au: np.ndarray          # AU, (n_stars, n_scenarios)
    outer_shift_au: np.ndarray
    width_shift_au: np.ndarray
    inner_shift_fraction: np.ndarray    # fraction of the reference boundary
    outer_shift_fraction: np.ndarray
    width_shift_fraction: np.ndarray
    shared_shift_fraction: np.ndarray   # (n_scenarios,) - the one relative shift every star agrees on, straight from the algebra
    star_to_star_spread: np.ndarray     # (n_scenarios,) - how far the per-star fractions actually drift apart


def sensitivity_metrics(boundaries: Boundaries, scenarios: Scenarios) -> Sensitivity:
    """
    gets the boundary table and the scenario list
    returns how far each scenario drags the edges away from the Earth-like reference -
    in AU and as a fraction of the reference distance

    inner_shift_au       = a_in - a_in,ref                 (in AU - same for a_out)
    inner_shift_fraction = (a_in - a_in,ref) / a_in,ref    (0.41 means 41% further out)
    width                = a_out - a_in                    so width_shift = width - width_ref

    the relative form is what lets an M dwarf and an F star be compared fairly - their
    zones sit at completely different distances so a shift of 0.1 AU means very different
    things to each
    """
    reference_index = scenarios.reference_index
    # the list index keeps the result a (n_stars, 1) column so it broadcasts against the full table
    reference_inner_edge = boundaries.a_in[:, [reference_index]]
    reference_outer_edge = boundaries.a_out[:, [reference_index]]
    reference_width = boundaries.width[:, [reference_index]]

    inner_shift_au = boundaries.a_in - reference_inner_edge
    outer_shift_au = boundaries.a_out - reference_outer_edge
    width_shift_au = boundaries.width - reference_width

    inner_shift_fraction = inner_shift_au / reference_inner_edge
    outer_shift_fraction = outer_shift_au / reference_outer_edge

    # A and eps reach a(T) only through scenarios.atmosphere_factor - so the relative shift is one number
    # per scenario that every star in the sample shares
    shared_shift_fraction = scenarios.atmosphere_factor / scenarios.atmosphere_factor[reference_index] - 1.0

    return Sensitivity(
        inner_shift_au=inner_shift_au,
        outer_shift_au=outer_shift_au,
        width_shift_au=width_shift_au,
        inner_shift_fraction=inner_shift_fraction,
        outer_shift_fraction=outer_shift_fraction,
        width_shift_fraction=width_shift_au / reference_width,
        shared_shift_fraction=shared_shift_fraction,
        star_to_star_spread=inner_shift_fraction.max(axis=0) - inner_shift_fraction.min(axis=0),
    )
