from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")   # TODO: explain what it is

import matplotlib.pyplot as plt
import numpy as np

from analysis import Analysis, CLASSES
from boundaries import Boundaries
from config import Config
from data import Sample
from model import hz_distance, planet_temp, star_radius
from scenarios import Scenarios
from sensitivity import Sensitivity

__all__ = ["make_all"]

CLASS_COLOUR = {"F": "#d1495b", "G": "#0077b6", "K": "#588157", "M": "#8338ec"}
ALBEDO_COLOUR = ("#588157", "#0077b6", "#d1495b")
GREENHOUSE_STYLE = ("-", "--", ":")
GUIDE_LINE = dict(color="0.4", linestyle=":", linewidth=1.2)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


@dataclass(frozen=True)
class _Sweep:
    """one atmosphere parameter running across its full range while the other holds still"""

    parameter_name: str             # "albedo" or "epsilon" - whichever one is moving
    swept_values: np.ndarray        # the range it moves over
    reference_value: float          # where it sits in the reference scenario
    other_fixed_value: float        # the other parameter - pinned at its own reference

    def atmosphere_arguments(self) -> dict:
        """what hz_distance expects - the moving parameter as an array, the other as one number"""
        other_name = "epsilon" if self.parameter_name == "albedo" else "albedo"
        return {self.parameter_name: self.swept_values, other_name: self.other_fixed_value}


def _build_sweeps(config: Config) -> tuple[_Sweep, _Sweep]:
    return (
        _Sweep("albedo", config.A_sweep, config.A_ref, config.eps_ref),
        _Sweep("epsilon", config.eps_sweep, config.eps_ref, config.A_ref),
    )


def _save(figure, config: Config, filename: str) -> None:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.figures_dir / filename)
    plt.close(figure)
    print(f"  wrote {config.figures_dir.name}/{filename}")


def _representative_stars(sample: Sample) -> dict[str, int]:
    """one star per class - the one sitting at that class's median luminosity"""
    representatives = {}
    for spectral_class in CLASSES:
        in_class = np.flatnonzero(sample.spec_class == spectral_class)
        median_luminosity = np.median(sample.L_sun[in_class])
        closest_to_median = np.argmin(np.abs(sample.L_sun[in_class] - median_luminosity))
        representatives[spectral_class] = int(in_class[closest_to_median])
    return representatives


def _boundaries_vs_atmosphere(sample: Sample, representatives: dict[str, int], config: Config) -> None:
    """figs 1 and 2 - both habitable zone edges of four representative stars as A and then eps sweep"""
    albedo_sweep, emissivity_sweep = _build_sweeps(config)
    panels = [
        (albedo_sweep, "albedo $A$",
         rf"HZ boundaries vs albedo   ($\epsilon = {config.eps_ref:.2f}$ fixed)",
         "fig1_boundaries_vs_albedo.png"),
        (emissivity_sweep, r"effective emissivity $\epsilon$   (smaller = stronger greenhouse)",
         rf"HZ boundaries vs greenhouse   ($A = {config.A_ref:.2f}$ fixed)",
         "fig2_boundaries_vs_greenhouse.png"),
    ]

    for sweep, x_label, title, filename in panels:
        atmosphere = sweep.atmosphere_arguments()
        figure, chart = plt.subplots(figsize=(9, 5.2))

        for spectral_class, star_index in representatives.items():
            colour = CLASS_COLOUR[spectral_class]
            star_teff = sample.teff[star_index]
            star_radius_m = sample.r_m[star_index]
            inner_edge_au = hz_distance(star_teff, star_radius_m, config.T_hot,
                                        **atmosphere) / config.AU
            outer_edge_au = hz_distance(star_teff, star_radius_m, config.T_cold,
                                        **atmosphere) / config.AU

            chart.fill_between(sweep.swept_values, inner_edge_au, outer_edge_au,
                               color=colour, alpha=0.15, linewidth=0)
            chart.plot(sweep.swept_values, inner_edge_au, "-", color=colour, linewidth=2,
                       label=rf"{spectral_class}   $T$={star_teff:.0f} K,  "
                             rf"$L$={sample.L_sun[star_index]:.3f} $L_\odot$")
            chart.plot(sweep.swept_values, outer_edge_au, "--", color=colour, linewidth=2)

        chart.axvline(sweep.reference_value, **GUIDE_LINE)
        chart.set_xlabel(x_label)
        chart.set_ylabel("distance from star  [AU]")
        chart.set_title(title + "\nsolid = $a_{in}$,  dashed = $a_{out}$", fontsize=11)
        chart.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        _save(figure, config, filename)


def _habitable_zone_width_map(sun_radius_m: float, config: Config) -> None:
    """fig 3 - how wide the habitable zone gets over the whole (A, eps) plane for a Sun-like star"""
    albedo_grid, emissivity_grid = np.meshgrid(config.A_sweep, config.eps_sweep)
    outer_edge = hz_distance(config.T_sun, sun_radius_m, config.T_cold, albedo_grid, emissivity_grid)
    inner_edge = hz_distance(config.T_sun, sun_radius_m, config.T_hot, albedo_grid, emissivity_grid)
    width_au = (outer_edge - inner_edge) / config.AU

    figure, chart = plt.subplots(figsize=(8.5, 5.4))
    shading = chart.contourf(albedo_grid, emissivity_grid, width_au, levels=24, cmap="viridis")
    contour_lines = chart.contour(albedo_grid, emissivity_grid, width_au, levels=np.arange(0.2, 1.3, 0.1), colors="black", linewidths=0.7)
    chart.clabel(contour_lines, fmt="%.1f", fontsize=9)
    chart.plot(config.A_ref, config.eps_ref, "o", markersize=11, markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.8)
    chart.annotate("reference", (config.A_ref, config.eps_ref), textcoords="offset points", xytext=(14, 0), verticalalignment="center", fontweight="bold")
    figure.colorbar(shading, ax=chart, label="HZ width  $a_{out} - a_{in}$  [AU]")
    chart.set_xlabel("albedo $A$")
    chart.set_ylabel(r"effective emissivity $\epsilon$")
    chart.set_title(r"HZ width for a Sun-like star ($L = 1\,L_\odot$)")
    chart.grid(False)
    _save(figure, config, "fig3_hz_width_map.png")


def _relative_sensitivity_collapse(sample: Sample, config: Config) -> None:
    """fig 4 - the relative shift is one single curve no matter which star you pick"""
    albedo_sweep, emissivity_sweep = _build_sweeps(config)
    figure, (left_chart, right_chart) = plt.subplots(1, 2, figsize=(11, 4.6))
    panels = [
        (left_chart, albedo_sweep, "albedo $A$"),
        (right_chart, emissivity_sweep, r"effective emissivity $\epsilon$"),
    ]

    for chart, sweep, x_label in panels:
        swept_distances = hz_distance(sample.teff[:, None], sample.r_m[:, None], config.T_hot, **sweep.atmosphere_arguments())
        reference_distances = hz_distance(sample.teff, sample.r_m, config.T_hot, config.A_ref, config.eps_ref)[:, None]
        relative_shift = swept_distances / reference_distances - 1.0
        widest_gap = np.ptp(relative_shift, axis=0).max()

        chart.plot(sweep.swept_values, 100 * relative_shift.min(axis=0), "-", color="#0077b6", linewidth=3.5, label="min over all stars")
        chart.plot(sweep.swept_values, 100 * relative_shift.max(axis=0), "--", color="#d1495b", linewidth=1.5, label="max over all stars")
        chart.axvline(sweep.reference_value, **GUIDE_LINE)
        chart.axhline(0, **GUIDE_LINE)
        chart.set_xlabel(x_label)
        chart.set_ylabel(r"$\Delta a\, /\, a_{ref}$   [%]")
        chart.set_title(f"spread over {sample.star_count} stars: {widest_gap:.1e}", fontsize=10)

    left_chart.legend(loc="lower left", fontsize=9)
    figure.suptitle("Relative sensitivity is identical for every star in the sample", fontsize=12)
    _save(figure, config, "fig4_relative_sensitivity_collapse.png")


def _habitable_zone_diagram(sample: Sample, boundaries: Boundaries, reference_index: int, config: Config) -> None:
    """fig 5 - where both edges actually fall for the real sample under the reference scenario"""
    figure, chart = plt.subplots(figsize=(9, 5.2))
    for spectral_class in CLASSES:
        in_class = sample.spec_class == spectral_class
        colour = CLASS_COLOUR[spectral_class]
        chart.scatter(sample.teff[in_class], boundaries.a_in[in_class, reference_index], s=4, color=colour, alpha=0.55, linewidth=0, label=f"{spectral_class}  (n={in_class.sum()})")
        chart.scatter(sample.teff[in_class], boundaries.a_out[in_class, reference_index], s=4, color=colour, alpha=0.55, linewidth=0)

    chart.set_yscale("log")
    chart.set_xlabel(r"$T_{eff}$  [K]")
    chart.set_ylabel("$a_{in}$  and  $a_{out}$   [AU]")
    chart.set_title(f"HZ boundaries for {sample.star_count} Gaia DR3 main-sequence stars (reference scenario)")
    chart.legend(markerscale=3, fontsize=9)
    _save(figure, config, "fig5_hz_diagram_sample.png")


def _shift_teff_vs_luminosity(sample: Sample, scenarios: Scenarios, sensitivity: Sensitivity, config: Config) -> None:
    """fig 6 - the same shifts twice - against Teff they scatter, against sqrt(L) they collapse"""
    strong_greenhouse = scenarios.names.index("midA_strongGH")
    inner_edge_shift_au = np.abs(sensitivity.inner_shift_au[:, strong_greenhouse])
    near_5000k = np.abs(sample.teff - 5000) < 100
    spread_factor = inner_edge_shift_au[near_5000k].max() / inner_edge_shift_au[near_5000k].min()

    figure, (left_chart, right_chart) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for spectral_class in CLASSES:
        in_class = sample.spec_class == spectral_class
        colour = CLASS_COLOUR[spectral_class]
        left_chart.scatter(sample.teff[in_class], inner_edge_shift_au[in_class], s=5, color=colour, alpha=0.5, linewidth=0)
        right_chart.scatter(np.sqrt(sample.L_sun[in_class]), inner_edge_shift_au[in_class], s=5, color=colour, alpha=0.5, linewidth=0, label=spectral_class)

    left_chart.set_yscale("log")
    left_chart.set_xlabel(r"$T_{eff}$  [K]")
    left_chart.set_ylabel(r"$|\Delta a_{in}|$  [AU]")
    left_chart.set_title(f"vs $T_{{eff}}$  -  {spread_factor:.0f}x spread inside "
                         f"$T_{{eff}}=5000\\pm100$ K", fontsize=10)
    right_chart.set_xlabel(r"$\sqrt{L / L_\odot}$")
    right_chart.set_ylabel(r"$|\Delta a_{in}|$  [AU]")
    right_chart.set_title("vs $\\sqrt{L}$  -  a single straight line", fontsize=10)
    right_chart.legend(markerscale=3, fontsize=9)
    figure.suptitle("Same shifts, two x-axes: the apparent $T_{eff}$ trend is luminosity", fontsize=12)
    _save(figure, config, "fig6_absolute_shift_teff_vs_L.png")


def _temperature_profiles(scenarios: Scenarios, sun_radius_m: float, config: Config) -> None:
    """fig 7 - Tp(a) for a Sun-like star under all nine scenarios, with the two water lines"""
    distance_grid_m = np.linspace(0.15, 2.6, 600) * config.AU

    figure, chart = plt.subplots(figsize=(9.5, 5.4))
    for scenario_index, (name, albedo, emissivity) in enumerate(scenarios):
        albedo_level = config.A_levels.index(albedo)
        greenhouse_level = config.eps_levels.index(emissivity)
        chart.plot(distance_grid_m / config.AU,
                   planet_temp(config.T_sun, sun_radius_m, distance_grid_m, albedo, emissivity),
                   GREENHOUSE_STYLE[greenhouse_level], color=ALBEDO_COLOUR[albedo_level],
                   linewidth=3.4 if scenario_index == scenarios.reference_index else 1.6, label=name)

    water_lines = [(config.T_hot, "$T_{hot}$ = 373 K"), (config.T_cold, "$T_{cold}$ = 273 K")]
    for temperature, label in water_lines:
        chart.axhline(temperature, color="0.35", linestyle=":", linewidth=1.2)
        chart.annotate(label, (0.17, temperature + 8), fontsize=9, color="0.35")

    chart.set_xlim(0.15, 2.6)
    chart.set_ylim(150, 500)
    chart.set_xlabel("distance $a$  [AU]")
    chart.set_ylabel("$T_p$  [K]")
    chart.set_title("$T_p(a)$ for a Sun-like star, all 9 scenarios\n"
                    "colour = albedo,  dash = greenhouse,  thick = reference", fontsize=11)
    chart.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    _save(figure, config, "fig7_temperature_profiles.png")


def _scenario_ranking(scenarios: Scenarios, sensitivity: Sensitivity, config: Config) -> None:
    """fig 8 - the nine scenarios sorted by how far they move both edges"""
    ranking = np.argsort(sensitivity.predicted_shift)
    shift_percent = 100 * sensitivity.predicted_shift[ranking]
    bar_positions = range(len(ranking))

    figure, chart = plt.subplots(figsize=(9, 4.8))
    chart.barh(bar_positions, shift_percent,
               color=["#d1495b" if shift < 0 else "#0077b6" for shift in shift_percent])
    chart.set_yticks(bar_positions, [scenarios.names[index] for index in ranking], fontsize=9)
    chart.axvline(0, color="black", linewidth=0.8)

    for position, shift in enumerate(shift_percent):
        chart.annotate(f"{shift:+.1f}%", (shift, position), xytext=(6 if shift >= 0 else -6, 0),
                       textcoords="offset points", verticalalignment="center",
                       horizontalalignment="left" if shift >= 0 else "right", fontsize=9)

    chart.set_xlim(-45, 72)
    chart.set_xlabel(r"$\Delta a\, /\, a_{ref}$   [%]      (identical for every star)")
    chart.set_title("Shift of both HZ boundaries, by scenario")
    _save(figure, config, "fig8_scenario_ranking.png")


def make_all(sample: Sample, scenarios: Scenarios, boundaries: Boundaries, sensitivity: Sensitivity, analysis: Analysis, config: Config) -> None:
    representatives = _representative_stars(sample)
    sun_radius_m = star_radius(config.L_sun, config.T_sun, config.sigma)

    _boundaries_vs_atmosphere(sample, representatives, config)
    _habitable_zone_width_map(sun_radius_m, config)
    _relative_sensitivity_collapse(sample, config)
    _habitable_zone_diagram(sample, boundaries, scenarios.reference_index, config)
    _shift_teff_vs_luminosity(sample, scenarios, sensitivity, config)
    _temperature_profiles(scenarios, sun_radius_m, config)
    _scenario_ranking(scenarios, sensitivity, config)
