from __future__ import annotations

import figures
import report
from analysis import analyze
from boundaries import compute_boundaries
from config import CONFIG
from data import load_gaia
from scenarios import build_scenarios
from sensitivity import sensitivity_metrics


def main() -> int:
    print("sample:")  # read Gaia DR3 - apply the quality cuts - derive R* from L and Teff
    sample = load_gaia(CONFIG)
    print(f"  {sample.cuts['kept']} of {sample.cuts['read']} stars retained")

    print("scenarios:")  # build the 3x3 grid of (albedo, greenhouse) atmospheres
    scenarios = build_scenarios(CONFIG)
    print(f"  {scenarios.scenario_count} scenarios, reference = {scenarios.names[scenarios.reference_index]}")

    print("boundaries:")  # solve a_in and a_out in AU for every star x scenario
    boundaries = compute_boundaries(sample, scenarios, CONFIG)
    print(f"  {sample.star_count} x {scenarios.scenario_count} boundary pairs")

    print("sensitivity:")  # how far each scenario drags the edges off the reference - in AU and as a fraction
    sensitivity = sensitivity_metrics(boundaries, scenarios)

    print("dependence on star type:")  # audit the claim that the edges track sqrt(L) alone and not Teff
    analysis = analyze(sample, boundaries, sensitivity, scenarios, CONFIG)

    print("writing tables:")  # dump the csv tables to the output dir
    report.write_tables(sample, scenarios, boundaries, sensitivity, analysis, CONFIG)

    print("writing figures:")  # render the plots and the text summary
    figures.make_all(sample, scenarios, boundaries, sensitivity, analysis, CONFIG)
    report.write_summary(sample, scenarios, boundaries, sensitivity, analysis, CONFIG)

    print(f"\ndone - see {CONFIG.out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
