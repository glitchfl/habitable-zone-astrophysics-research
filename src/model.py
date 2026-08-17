""" The main equations
    R*    = sqrt( L / (4 pi sigma Teff^4) )               Stefan-Boltzmann, for radius
    S(a)  = L / (4 pi a^2) = sigma Teff^4 (R*/a)^2        inverse square, two equal forms
            (1-A) S(a)/4 = eps sigma Tp^4                 energy balance
    Tp(a) = Teff ((1-A)/(4 eps))^(1/4) (R*/a)^(1/2)       forward
    a(T)  = R* ((1-A)/(4 eps))^(1/2) (Teff/T)^2           inverse - the working formula

Everything is numpy-vectorised (meaning it can take arrays as arguments) so any argument may be a scalar or an array as long as
the shapes broadcast (that is what lets `hz_distance` fill a whole (star x scenario) grid in one call)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["star_radius", "stellar_flux", "atmosphere_factor", "planet_temp", "hz_distance"]


def star_radius(L_watt: ArrayLike, teff: ArrayLike, sigma: float) -> np.ndarray:
    """
    gets L (in Watts) T_eff (in kelvin) and the stefan boltzmann constant (sigma)
    returns the radius of the star (in meters)
    R_* = sqrt( L / 4 pi * sigma * T_eff^4)
    """
    return np.sqrt(np.asarray(L_watt) / (4.0 * np.pi * sigma * np.asarray(teff) ** 4))


def stellar_flux(L_watt: ArrayLike, distance_m: ArrayLike) -> np.ndarray:
    """
    gets L (in Watts) and a distance a from the star (in meters)
    returns the radiation flux arriving at that distance (in W/m^2)
    S(a) = L / (4 pi a^2)

    the star's whole output spreads over a sphere of radius a - so doubling the distance
    spreads it over 4 times the area and the flux drops by 4
    """
    return np.asarray(L_watt) / (4.0 * np.pi * np.asarray(distance_m) ** 2)


def atmosphere_factor(albedo: ArrayLike, epsilon: ArrayLike) -> np.ndarray:
    """
    gets the albedo A (0 to 1 - the share of starlight the planet reflects straight back)
    and epsilon (0 to 1 - how freely the planet radiates its heat to space)
    returns the dimensionless number sqrt((1-A) / (4 eps))

    this one factor is the *only* way the atmosphere reaches a(T) - a scenario multiplies
    both edges by it and does nothing else
    """
    return np.sqrt((1.0 - np.asarray(albedo)) / (4.0 * np.asarray(epsilon)))


def planet_temp(teff: ArrayLike, r_star_m: ArrayLike, distance_m: ArrayLike, albedo: ArrayLike, epsilon: ArrayLike) -> np.ndarray:
    """
    gets T_eff (in kelvin) the star radius R_* (in meters) the distance a (in meters) and the albedo A and epsilon
    returns the planet's avg temperature once it settles (in kelvin)
    T_p(a) = T_eff * ((1-A)/(4 eps))^(1/4) * sqrt(R_* / a)

    this is the energy balance (1-A) S(a) / 4 = eps sigma T_p^4 solved for T_p - the
    planet keeps warming until what it radiates matches what it absorbs

    the 1/4 is geometry - the planet catches starlight on a flat disc of area pi R_p^2
    but radiates from its whole surface 4 pi R_p^2
    """
    return (np.asarray(teff) * atmosphere_factor(albedo, epsilon) ** 0.5 * np.sqrt(np.asarray(r_star_m) / np.asarray(distance_m)))


def hz_distance(teff: ArrayLike, r_star_m: ArrayLike, target_temp: ArrayLike, albedo: ArrayLike, epsilon: ArrayLike) -> np.ndarray:
    """
    gets T_eff (in kelvin) the star radius R_* (in meters) a wanted temperature target_temp (in kelvin) the albedo A and epsilon
    returns the distance from the star where the planet would sit at that temperature (in meters)
    a(T) = R_* * sqrt((1-A)/(4 eps)) * (T_eff / T)^2
    """
    return (np.asarray(r_star_m) * atmosphere_factor(albedo, epsilon) * (np.asarray(teff) / np.asarray(target_temp)) ** 2)
