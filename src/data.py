"""read the Gaia DR3 rows, apply quality cuts and derive R* from L and Teff"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config
from model import star_radius

EXPECTED_COLUMNS = [
    "SOURCE_ID", "teff_gspphot", "lum_flame", "radius_flame",
    "parallax_over_error", "ruwe", "spectral",
]


@dataclass
class Sample:
    """The retained stars (with everything the model needs already in SI)"""

    source_id: np.ndarray      # str (see note in `load_gaia`)
    spec_class: np.ndarray     # 'F' / 'G' / 'K' / 'M'
    teff: np.ndarray           # K
    L_sun: np.ndarray          # solar luminosities
    L_watt: np.ndarray         # W
    r_gaia_rsun: np.ndarray    # Gaia's own radius_flame (for cross-checking)
    r_m: np.ndarray            # our R* (metres)
    r_rsun: np.ndarray         # our R* (solar radii)
    cuts: dict[str, int]

    @property
    def star_count(self) -> int:
        return len(self.teff)

    @property
    def radius_residual(self) -> np.ndarray:
        """
        returns the fractional gap between our R_* and the radius Gaia publishes itself
        (0.01 would mean ours is 1% larger)

        (Gaia derives radius_flame through a completely separate pipeline)
        """
        return self.r_rsun / self.r_gaia_rsun - 1.0


def load_gaia(config: Config) -> Sample:
    """
    gets the config (which holds the csv path and every quality cut)
    returns the stars that survive the cuts - with L (in watts) and R_* (in meters)

    reads gaia dr3 - drops rows with missing or impossible values - keep only well measured
    single main-sequence stars in the T_eff window

    NOTE: SOURCE_ID is read as text to avoid float conversion and scientific notation issues
    """
    catalogue = pd.read_csv(config.csv_file, dtype={"SOURCE_ID": str}) # SOURCE_ID is read as text to avoid float conversion and scientific notation issues

    # check column names
    if list(catalogue.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"unexpected columns\n  got:  {list(catalogue.columns)}\n  want: {EXPECTED_COLUMNS}"
        )

    # extract columns
    teff = catalogue["teff_gspphot"].to_numpy(float)
    luminosity_lsun = catalogue["lum_flame"].to_numpy(float)
    radius_rsun = catalogue["radius_flame"].to_numpy(float)
    parallax_over_error = catalogue["parallax_over_error"].to_numpy(float)
    ruwe = catalogue["ruwe"].to_numpy(float)

    # define the cuts - going over 1d array one by one
    is_finite = np.isfinite(teff) & np.isfinite(luminosity_lsun) & np.isfinite(radius_rsun) & np.isfinite(parallax_over_error) & np.isfinite(ruwe)
    is_positive = (teff > 0) & (luminosity_lsun > 0) & (radius_rsun > 0)
    passes_astrometry = (parallax_over_error >= config.min_parallax_over_error) & (ruwe <= config.max_ruwe)
    in_teff_window = (teff >= config.teff_min) & (teff <= config.teff_max)
    is_main_sequence = radius_rsun <= config.max_radius_rsun

    keep = is_finite & is_positive & passes_astrometry & in_teff_window & is_main_sequence

    # each cut counted against everything that survived the previous ones
    cuts = {
        "read": len(catalogue),
        "non_finite": int((~is_finite).sum()),
        "non_positive": int((is_finite & ~is_positive).sum()),
        "astrometry": int((is_finite & is_positive & ~passes_astrometry).sum()),
        "teff_window": int((is_finite & is_positive & passes_astrometry & ~in_teff_window).sum()),
        "giants": int((is_finite & is_positive & passes_astrometry & in_teff_window & ~is_main_sequence).sum()),
        "kept": int(keep.sum()),
    }

    kept_luminosity_lsun = luminosity_lsun[keep]
    kept_teff = teff[keep]
    L_watt = kept_luminosity_lsun * config.L_sun
    r_m = star_radius(L_watt, kept_teff, config.sigma) # our calculation for the star radius

    return Sample(
        source_id=catalogue["SOURCE_ID"].to_numpy()[keep],
        spec_class=catalogue["spectral"].str[0].to_numpy()[keep],
        teff=kept_teff,
        L_sun=kept_luminosity_lsun,
        L_watt=L_watt,
        r_gaia_rsun=radius_rsun[keep],
        r_m=r_m,
        r_rsun=r_m / config.R_sun,
        cuts=cuts,
    )
