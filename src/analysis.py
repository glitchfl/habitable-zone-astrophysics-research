"""does the sensitivity depend on the kind of star?

Substituting R* = sqrt(L / (4 pi sigma Teff^4)) into a(T) cancels Teff exactly:

    a(T) = R* sqrt((1-A)/(4 eps)) (Teff/T)^2
         = sqrt(L/(4 pi sigma)) / Teff^2 * sqrt((1-A)/(4 eps)) * Teff^2 / T^2
         = sqrt(L/(4 pi sigma)) * sqrt((1-A)/(4 eps)) / T^2

so inside this model the boundaries know the star only through sqrt(L) - everything here
is the numerical audit of that statement against the real sample
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from boundaries import Boundaries
from config import Config
from data import Sample
from model import atmosphere_factor, hz_distance, star_radius
from scenarios import Scenarios
from sensitivity import Sensitivity

CLASSES = "FGKM"


@dataclass
class Analysis:
    """each field is a separate test of the same claim - that the edges depend on the
    star only through sqrt(L) and not on T_eff at all"""

    collapse_residual: float                # worst gap between a_in and the Teff-free formula - should be ~0
    loglog_slope: float                     # slope of log a_in against log L - should be exactly 0.5
    loglog_residual: float                  # worst departure from that straight line
    correlation_a_in_with_teff: float       # raw correlation of a_in with Teff - looks like a Teff effect
    correlation_a_in_with_log_L: float      # raw correlation of a_in with log L
    residual_std_a_in: float                # scatter left in a_in once L is removed - numerically zero
    residual_std_teff: float                # scatter left in Teff once L is removed
    twin_count: int                         # how many real Gaia stars share the median luminosity
    twin_luminosity_lsun: float             # that luminosity (in L_sun)
    twin_teff_span: float                   # how far apart those twins Teff spread (in K)
    twin_a_in_spread_fraction: float        # how far apart their a_in spread as a fraction - should be ~0
    probe_teff: np.ndarray                  # three made-up stars given identical L and very different Teff
    probe_a_in: np.ndarray                  # their inner edges (in AU) - should be the same number
    probe_spread: float                     # the gap between those (in AU) - should be ~0
    per_class: dict = field(default_factory=dict)   # medians for F G K M separately


def analyze(sample: Sample, boundaries: Boundaries, sensitivity: Sensitivity, scenarios: Scenarios, config: Config) -> Analysis:
    """
    gets the sample the boundary table the sensitivity table the scenarios and the config
    returns the numbers that answer is the sensitivity different for different
    kinds of star
    """
    reference_index = scenarios.reference_index
    reference_a_in = boundaries.a_in[:, reference_index]

    # test 1 - the Teff-free form of a(T) computed straight from L - if Teff really cancels this has to reproduce a_in exactly
    teff_free_a_in = (np.sqrt(sample.L_watt / (4 * np.pi * config.sigma)) * atmosphere_factor(config.A_ref, config.eps_ref) / config.T_hot**2 / config.AU)
    collapse_residual = float(np.abs(teff_free_a_in / reference_a_in - 1).max())

    # test 2 - a goes as sqrt(L) so plotting log a_in against log L must give a perfectly straight line of slope 1/2
    log_luminosity = np.log(sample.L_sun)
    log_a_in = np.log(reference_a_in)
    slope, intercept = np.polyfit(log_luminosity, log_a_in, 1)
    log_a_in_residual = log_a_in - (slope * log_luminosity + intercept)

    # test 3 - a_in and Teff correlate at +0.90 which looks like a Teff effect - so strip
    # log L out of both and see what Teff still has to explain - the a_in residual comes
    # out numerically zero, so this is not a "weak" dependence - there is no variance left
    # for Teff to act on at all
    teff_slope, teff_intercept = np.polyfit(log_luminosity, sample.teff, 1)
    teff_residual = sample.teff - (teff_slope * log_luminosity + teff_intercept)

    # test 4 - real Gaia stars that happen to share a luminosity - a control group taken
    # from the data rather than from the algebra - same L should mean same edges however
    # far apart their Teff are
    median_luminosity = float(np.median(sample.L_sun))
    twins = np.abs(sample.L_sun / median_luminosity - 1) < 0.01
    twin_a_in = reference_a_in[twins]

    # test 5 - three invented stars given identical L and wildly different Teff - the
    # cleanest possible version of the experiment
    probe_teff = np.array([3200.0, 5772.0, 7400.0])
    probe_a_in = np.array([
        hz_distance(temperature, star_radius(config.L_sun, temperature, config.sigma), config.T_hot, config.A_ref, config.eps_ref) / config.AU
        for temperature in probe_teff
    ])

    per_class = {}
    for spectral_class in CLASSES:
        in_class = sample.spec_class == spectral_class
        per_class[spectral_class] = {
            "star_count": int(in_class.sum()),
            "median_teff": float(np.median(sample.teff[in_class])),
            "median_L": float(np.median(sample.L_sun[in_class])),
            "median_a_in": float(np.median(boundaries.a_in[in_class, reference_index])),
            "median_a_out": float(np.median(boundaries.a_out[in_class, reference_index])),
            "median_abs_inner_shift": np.median(np.abs(sensitivity.inner_shift_au[in_class, :]), axis=0),
        }

    return Analysis(
        collapse_residual=collapse_residual,
        loglog_slope=float(slope),
        loglog_residual=float(np.abs(log_a_in_residual).max()),
        correlation_a_in_with_teff=float(np.corrcoef(sample.teff, reference_a_in)[0, 1]),
        correlation_a_in_with_log_L=float(np.corrcoef(log_luminosity, log_a_in)[0, 1]),
        residual_std_a_in=float(log_a_in_residual.std()),
        residual_std_teff=float(teff_residual.std()),
        twin_count=int(twins.sum()),
        twin_luminosity_lsun=median_luminosity,
        twin_teff_span=float(np.ptp(sample.teff[twins])),
        twin_a_in_spread_fraction=float((twin_a_in.max() - twin_a_in.min()) / twin_a_in.mean()),
        probe_teff=probe_teff,
        probe_a_in=probe_a_in,
        probe_spread=float(np.ptp(probe_a_in)),
        per_class=per_class,
    )
