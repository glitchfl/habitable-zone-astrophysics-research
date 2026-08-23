from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import Analysis, CLASSES
from boundaries import Boundaries
from config import Config
from data import Sample
from scenarios import Scenarios
from sensitivity import Sensitivity


def star_table(sample: Sample) -> pd.DataFrame:
    return pd.DataFrame({
        "source_id": sample.source_id,
        "spec_class": sample.spec_class,
        "teff_K": sample.teff,
        "L_Lsun": sample.L_sun,
        "R_gaia_Rsun": sample.r_gaia_rsun,
        "R_derived_Rsun": sample.r_rsun,
        "R_resid_pct": 100 * sample.radius_residual,
    })


def long_table(sample: Sample, scenarios: Scenarios, boundaries: Boundaries, sensitivity: Sensitivity) -> pd.DataFrame:
    """one row per star per scenario - boundaries and shifts side by side"""
    star_count, scenario_count = boundaries.a_in.shape
    return pd.DataFrame({
        "source_id": np.tile(sample.source_id, scenario_count),
        "spec_class": np.tile(sample.spec_class, scenario_count),
        "teff_K": np.tile(sample.teff, scenario_count),
        "L_Lsun": np.tile(sample.L_sun, scenario_count),
        "scenario": np.repeat(scenarios.names, star_count),
        "A": np.repeat(scenarios.albedo, star_count),
        "eps": np.repeat(scenarios.epsilon, star_count),
        "a_in_AU": boundaries.a_in.T.ravel(),
        "a_out_AU": boundaries.a_out.T.ravel(),
        "width_AU": boundaries.width.T.ravel(),
        "d_a_in_AU": sensitivity.inner_shift_au.T.ravel(),
        "d_a_out_AU": sensitivity.outer_shift_au.T.ravel(),
        "d_width_AU": sensitivity.width_shift_au.T.ravel(),
        "d_a_in_pct": 100 * sensitivity.inner_shift_fraction.T.ravel(),
        "d_a_out_pct": 100 * sensitivity.outer_shift_fraction.T.ravel(),
        "d_width_pct": 100 * sensitivity.width_shift_fraction.T.ravel(),
    })


def scenario_table(scenarios: Scenarios, boundaries: Boundaries, sensitivity: Sensitivity) -> pd.DataFrame:
    return pd.DataFrame({
        "scenario": scenarios.names,
        "A": scenarios.albedo,
        "eps": scenarios.epsilon,
        "rel_shift_pct": 100 * sensitivity.predicted_shift,
        "med_a_in_AU": np.median(boundaries.a_in, axis=0),
        "med_a_out_AU": np.median(boundaries.a_out, axis=0),
        "med_width_AU": np.median(boundaries.width, axis=0),
        "med_abs_d_a_in_AU": np.median(np.abs(sensitivity.inner_shift_au), axis=0),
        "star_to_star_spread": sensitivity.star_to_star_spread,
    })


def class_table(analysis: Analysis, scenarios: Scenarios) -> pd.DataFrame:
    rows = []
    for spectral_class, class_stats in analysis.per_class.items():
        row = {"spec_class": spectral_class, "n": class_stats["star_count"],
               "med_teff_K": class_stats["median_teff"], "med_L_Lsun": class_stats["median_L"],
               "med_a_in_AU": class_stats["median_a_in"], "med_a_out_AU": class_stats["median_a_out"]}
        row.update({f"med_abs_d_a_in_{scenario_name}": median_shift
                    for scenario_name, median_shift
                    in zip(scenarios.names, class_stats["median_abs_inner_shift"])})
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(sample: Sample, scenarios: Scenarios, boundaries: Boundaries, sensitivity: Sensitivity, analysis: Analysis, config: Config) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table, number_format in [
        ("sample_stars.csv", star_table(sample), "%.8g"),
        ("hz_results.csv", long_table(sample, scenarios, boundaries, sensitivity), "%.8g"),
        ("scenario_summary.csv", scenario_table(scenarios, boundaries, sensitivity), "%.8g"),
        ("class_summary.csv", class_table(analysis, scenarios), "%.8g"),
    ]:
        table.to_csv(config.out_dir / filename, index=False, float_format=number_format)
        print(f"  wrote {filename}  ({len(table)} rows)")
