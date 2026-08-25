from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    csv_file: Path = ROOT / "csv" / "gaia_dr3_data.csv"
    out_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "results" / "figures"
    tables_dir: Path = ROOT / "results" / "tables"

    # SI constants - CODATA 2018 sigma, IAU 2015 nominal solar values
    sigma: float = 5.670374419e-8    # W m^-2 K^-4 - CODATA 2018 - exact since the 2019 SI redefinition
    L_sun_watt: float = 3.828e26     # W - IAU 2015 resolution B3 nominal
    R_sun_m: float = 6.957e8         # m - IAU 2015 resolution B3 nominal
    AU: float = 1.495978707e11       # m - IAU 2012 resolution B2 - exact by definition
    T_sun: float = 5772.0            # K - IAU 2015 resolution B3 nominal - only feeds the Sun/Earth anchors

    # quality cuts
    min_parallax_over_error: float = 20.0        # distance good to 5% (L goes as d^2 and a goes as sqrt(L) so the edges land within 5%) || TODO: tune
    max_ruwe: float = 1.4                        # isolated single stars (Lindegren 2018 cut - above it the fit is likely an unresolved binary) || TODO: tune
    max_radius_rsun: float = 2.0                 # drops giants || TODO: tune
    teff_min: float = 2800.0                     # M-stars
    teff_max: float = 8000.0                     # F-stars

    # liquid-water thresholds at 1 atm
    T_hot: float = 373.15                        # 100°C
    T_cold: float = 273.15                       # 0°C

    # atmosphere scenarios (low/mid/high albedo x weak/mid/strong greenhouse)
    albedo_levels: tuple = (0.10, 0.30, 0.50)
    epsilon_levels: tuple = (1.00, 0.60, 0.30)   # descending - eps=1 radiates freely (no greenhouse), eps=0.30 traps the most heat

    # reference - A is Earth's Bond albedo, eps=0.6 reproduces Earth's 288 K surface
    albedo_ref: float = 0.30
    epsilon_ref: float = 0.60

    # fine sweeps for the sensitivity curves (part 4)
    albedo_sweep: np.ndarray = field(default_factory=lambda: np.linspace(0.00, 0.80, 81))
    epsilon_sweep: np.ndarray = field(default_factory=lambda: np.linspace(0.20, 1.00, 81))


CONFIG = Config()
