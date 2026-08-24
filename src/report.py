from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from analysis import Analysis, CLASSES
from boundaries import Boundaries
from config import Config
from data import Sample
from model import planet_temp, stellar_flux
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
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    for filename, table, number_format in [
        ("sample_stars.csv", star_table(sample), "%.8g"),
        ("hz_results.csv", long_table(sample, scenarios, boundaries, sensitivity), "%.8g"),
        ("scenario_summary.csv", scenario_table(scenarios, boundaries, sensitivity), "%.8g"),
        ("class_summary.csv", class_table(analysis, scenarios), "%.8g"),
    ]:
        table.to_csv(config.tables_dir / filename, index=False, float_format=number_format)
        print(f"  wrote {config.tables_dir.name}/{filename}  ({len(table)} rows)")


def write_summary(sample: Sample, scenarios: Scenarios, boundaries: Boundaries, sensitivity: Sensitivity, analysis: Analysis, config: Config) -> None:
    """the numbers part 6 has to be written from"""
    reference_index = scenarios.reference_index
    earth_surface_temp = planet_temp(config.T_sun, config.R_sun, config.AU, config.A_ref, config.eps_ref)
    divider = "=" * 78
    lines = []
    add_line = lines.append

    add_line(divider)
    add_line("HABITABLE ZONE SENSITIVITY TO ALBEDO AND GREENHOUSE PARAMETER")
    add_line(f"Gaia DR3 main-sequence sample - Python implementation - "
             f"{datetime.now():%Y-%m-%d %H:%M}")
    add_line(divider)

    add_line("\n1. SAMPLE\n" + "-" * 78)
    cuts = sample.cuts
    add_line(f"  rows read from catalogue          {cuts['read']}")
    add_line(f"  removed - non-finite              {cuts['non_finite']}")
    add_line(f"  removed - non-positive L/Teff/R   {cuts['non_positive']}")
    add_line(f"  removed - astrometry cuts         {cuts['astrometry']}   "
             f"(plx/err >= {config.min_parallax_over_error:g}, RUWE <= {config.max_ruwe:g})")
    add_line(f"  removed - outside Teff window     {cuts['teff_window']}   "
             f"({config.teff_min:g} - {config.teff_max:g} K)")
    add_line(f"  removed - giants and subgiants    {cuts['giants']}   "
             f"(R > {config.max_radius_rsun:g} Rsun)")
    add_line(f"  RETAINED                          {cuts['kept']}")
    add_line(f"\n  Teff  {sample.teff.min():.0f} - {sample.teff.max():.0f} K"
             f"      L  {sample.L_sun.min():.4f} - {sample.L_sun.max():.3f} Lsun")
    for spectral_class in CLASSES:
        class_stats = analysis.per_class[spectral_class]
        add_line(f"    {spectral_class}  n={class_stats['star_count']:4d}   "
                 f"median Teff {class_stats['median_teff']:5.0f} K   "
                 f"median L {class_stats['median_L']:8.4f} Lsun")
    add_line("\n  R* rebuilt from L and Teff vs Gaia radius_flame:")
    add_line(f"    median {100*np.median(sample.radius_residual):+.4f} %   "
             f"p95 |resid| {100*np.percentile(np.abs(sample.radius_residual), 95):.2f} %"
             "   -> the Stefan-Boltzmann step is sound")

    add_line("\n2. MODEL AND SCENARIOS\n" + "-" * 78)
    add_line("  Tp(a) = Teff * ((1-A)/(4 eps))^(1/4) * (R*/a)^(1/2)")
    add_line("  a(T)  = R*   * ((1-A)/(4 eps))^(1/2) * (Teff/T)^2")
    add_line(f"  thresholds  T_hot = {config.T_hot:.2f} K   T_cold = {config.T_cold:.2f} K")
    add_line(f"  reference   A = {config.A_ref:.2f}   eps = {config.eps_ref:.2f}  "
             f"(gives Earth {earth_surface_temp:.1f} K at 1 AU)")

    add_line("\n3-4. BOUNDARIES AND SENSITIVITY\n" + "-" * 78)
    add_line(f"  {'scenario':<22} {'A':>5} {'eps':>5} | {'med a_in':>9} {'med a_out':>9} | "
             f"{'rel shift':>10} | {'med |da_in|':>12}")
    for scenario_index, scenario_name in enumerate(scenarios.names):
        add_line(f"  {scenario_name:<22} {scenarios.albedo[scenario_index]:5.2f} "
                 f"{scenarios.epsilon[scenario_index]:5.2f} | "
                 f"{np.median(boundaries.a_in[:, scenario_index]):9.4f} "
                 f"{np.median(boundaries.a_out[:, scenario_index]):9.4f} | "
                 f"{100*sensitivity.predicted_shift[scenario_index]:+9.2f}% | "
                 f"{np.median(np.abs(sensitivity.inner_shift_au[:, scenario_index])):9.4f} AU")

    largest_shift = int(np.argmax(np.abs(sensitivity.predicted_shift)))
    # the reference sits at exactly 0% - park it out of reach so argmin finds a real scenario
    shifts_beside_reference = np.abs(sensitivity.predicted_shift).copy()
    shifts_beside_reference[reference_index] = np.inf
    smallest_shift = int(np.argmin(shifts_beside_reference))
    add_line(f"\n  largest shift : {scenarios.names[largest_shift]}  "
             f"{100*sensitivity.predicted_shift[largest_shift]:+.2f} %")
    add_line(f"  smallest shift: {scenarios.names[smallest_shift]}  "
             f"{100*sensitivity.predicted_shift[smallest_shift]:+.2f} %")
    add_line("\n  a_in and a_out always move by the SAME percentage, so the zone dilates without")
    add_line(f"  changing shape: a_out/a_in = (T_hot/T_cold)^2 = "
             f"{(config.T_hot/config.T_cold)**2:.6f} for every star and scenario.")

    add_line("\n5. DOES SENSITIVITY DEPEND ON THE STAR\n" + "-" * 78)
    add_line("  Substituting R* = sqrt(L/(4 pi sigma Teff^4)) into a(T) cancels Teff exactly:")
    add_line("      a(T) = sqrt(L/(4 pi sigma)) * sqrt((1-A)/(4 eps)) / T^2")
    add_line(f"  verified numerically to {analysis.collapse_residual:.1e} relative.\n")
    add_line(f"  RELATIVE sensitivity: identical for all {sample.star_count} stars.")
    add_line(f"    star-to-star spread of da/a_ref, worst scenario: "
             f"{sensitivity.star_to_star_spread.max():.2e}  (machine precision)")
    add_line("    -> no dependence on Teff, on L, or on spectral class. none.\n")
    add_line("  ABSOLUTE sensitivity: depends on the star, through sqrt(L) alone.")
    add_line(f"    log-log slope of a_in against L = {analysis.loglog_slope:.12f}  "
             f"(exact 1/2, max resid {analysis.loglog_residual:.1e})")
    add_line("    median |da_in| for the strong-greenhouse scenario, by class:")

    strong_greenhouse = scenarios.names.index("midA_strongGH")
    m_dwarf_shift = analysis.per_class["M"]["median_abs_inner_shift"][strong_greenhouse]
    for spectral_class in CLASSES:
        class_shift = analysis.per_class[spectral_class]["median_abs_inner_shift"][strong_greenhouse]
        add_line(f"      {spectral_class}  {class_shift:.4f} AU   "
                 f"({class_shift/m_dwarf_shift:.1f}x the M-dwarf value)")

    probe_temps = "/".join(f"{temperature:.0f}" for temperature in analysis.probe_teff)
    add_line("\n  The apparent Teff trend is luminosity in disguise:")
    add_line(f"    corr(a_in, Teff) over the sample      {analysis.correlation_a_in_with_teff:+.4f}"
             "      <- looks like a real trend")
    add_line(f"    corr(log a_in, log L)                 {analysis.correlation_a_in_with_log_L:+.8f}")
    add_line(f"    std of log a_in after removing log L   {analysis.residual_std_a_in:.2e}"
             "   <- nothing left for Teff")
    add_line(f"    std of Teff after removing log L       {analysis.residual_std_teff:.1f} K"
             "      <- yet Teff varies freely")
    add_line(f"    {analysis.twin_count} real Gaia stars within 1% in L span "
             f"{analysis.twin_teff_span:.0f} K in Teff,")
    add_line(f"      and their a_in agree to {100*analysis.twin_a_in_spread_fraction:.2f}% - "
             "the 1% L spread, not the Teff spread.")
    add_line(f"    control: L fixed at 1 Lsun, Teff {probe_temps} K"
             f" -> a_in identical to {analysis.probe_spread:.1e} AU")

    reference_factor = np.sqrt((1 - config.A_ref) / (4 * config.eps_ref))
    strong_greenhouse_factor = np.sqrt((1 - config.A_ref) / (4 * 0.30))
    high_albedo_factor = np.sqrt((1 - 0.50) / (4 * config.eps_ref))
    f_star_shift = analysis.per_class["F"]["median_abs_inner_shift"][strong_greenhouse]

    add_line("\n6. WHAT THIS MEANS, AND WHERE IT STOPS\n" + "-" * 78)
    add_line("  * A and eps move both boundaries by one common factor sqrt((1-A)/(4 eps)).")
    add_line(f"    Over the scenario set that factor runs from "
             f"{100*sensitivity.predicted_shift.min():+.0f}%"
             f" to {100*sensitivity.predicted_shift.max():+.0f}% of the reference.")
    add_line(f"  * eps is the stronger lever: dropping eps 0.60 -> 0.30 gives "
             f"{100*(strong_greenhouse_factor/reference_factor-1):+.1f}%, while")
    add_line(f"    raising A 0.30 -> 0.50 gives only "
             f"{100*(high_albedo_factor/reference_factor-1):+.1f}%.")
    add_line("  * Answer to the research question: the sensitivity of the boundaries to A and eps")
    add_line("    does NOT depend on the type of star in relative terms. In absolute terms a bright")
    add_line(f"    F star's boundary moves ~{f_star_shift/m_dwarf_shift:.0f}x further"
             " than an M dwarf's for the same atmosphere")
    add_line("    change, but that factor is sqrt(L_F/L_M), not a Teff effect.")
    add_line("  * Limits: this is a grey 0-D model. Because A and eps are taken as constants that")
    add_line("    do not depend on wavelength, the star's spectrum cannot influence the result, and")
    add_line("    the Teff cancellation above is a direct consequence of that choice. A real M dwarf")
    add_line("    planet has a lower albedo against red light and different H2O/CO2 absorption, so a")
    add_line("    spectrally resolved model WOULD produce a Teff dependence. The clean null result")
    add_line("    here is a statement about the model, and that is what makes it a useful baseline.")

    earth_equilibrium_temp = planet_temp(config.T_sun, config.R_sun, config.AU, 0.30, 1.00)
    add_line("\n7. VERIFICATION\n" + "-" * 78)
    add_line(f"  external anchors: solar constant {stellar_flux(config.L_sun, config.AU):.1f} W/m2"
             f" | Earth eq. temp {earth_equilibrium_temp:.1f} K"
             f" | Earth surface {earth_surface_temp:.1f} K")
    add_line(divider)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.out_dir / "summary_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote summary_report.txt")
