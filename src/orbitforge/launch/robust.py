"""Scenario-based robust launch window screening.

`orbitforge.launch.windows` answers *when* the plane is reachable and what it
costs on the nominal date. This module answers whether the plan still closes
when the launch slips, the injection is off-nominal and ground stations drop
out: the three stresses are evaluated as one joint scenario set, and a plan is
robust only if at least `required_fraction` of the scenarios satisfy the fuel,
power and communications constraints simultaneously.

Screening-level models, by design:
  * a slipped launch waits for the next plane crossing, so delay leaves ascent
    energy unchanged but degrades the eclipse margin linearly (seasonal
    geometry) via `eclipse_margin_decay_wh_per_day`;
  * injection errors (da, di) are corrected by the spacecraft with a Hohmann
    transfer plus a plane change, charged against its propellant;
  * a station outage removes that station's passes for the day;
  * date-dependent effects the screening model cannot derive (e.g. from a
    higher-fidelity ephemeris) can be injected per plan date through the
    mission's `plan_overrides` mapping.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from itertools import combinations

from orbitforge.core.constants import DAY_S, MU_EARTH_KM3_S2
from orbitforge.launch.ascent_budget import delta_v_budget, ideal_orbital_speed
from orbitforge.launch.windows import azimuth_for_inclination, launch_energy_bonus_km_s, plane_crossing_times
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.maneuvers.plane_change import plane_change_delta_v
from orbitforge.propulsion.budget import propellant_required
from orbitforge.signoff import contact_quality, power_quality, propellant_quality


@dataclass(frozen=True)
class Scenario:
    delay_days: float
    insertion_da_km: float
    insertion_di_rad: float
    stations_down: tuple = ()


def insertion_error_points(sigma_a_km, sigma_i_rad, n_sigma=3.0):
    a = n_sigma * sigma_a_km
    i = n_sigma * sigma_i_rad
    return [(0.0, 0.0), (a, 0.0), (-a, 0.0), (0.0, i), (0.0, -i), (a, i), (a, -i), (-a, i), (-a, -i)]


def station_outage_sets(stations, max_down=1):
    out = [()]
    for k in range(1, max_down + 1):
        out.extend(combinations(stations, k))
    return out


def build_scenarios(delays_days, insertion_errors, station_outages):
    return [Scenario(d, da, di, down) for d in delays_days for da, di in insertion_errors for down in station_outages]


def insertion_recovery_delta_v_km_s(radius_km, da_km, di_rad, mu_km3_s2=MU_EARTH_KM3_S2):
    orbit_dv = hohmann(radius_km, radius_km + da_km, mu_km3_s2)['total_dv_km_s']
    plane_dv = plane_change_delta_v(ideal_orbital_speed(mu_km3_s2, radius_km), di_rad)
    return orbit_dv + plane_dv


def _ascent(launch, orbit, inclination_rad):
    azimuth = azimuth_for_inclination(launch['site_lat_rad'], inclination_rad)
    bonus = launch_energy_bonus_km_s(launch['site_lat_rad'], azimuth)
    mu = orbit.get('mu_km3_s2', MU_EARTH_KM3_S2)
    budget = delta_v_budget(ideal_orbital_speed(mu, orbit['radius_km']), bonus, **launch.get('ascent_losses_km_s', {}))
    return {
        'azimuth_rad': azimuth,
        'rotation_bonus_km_s': bonus,
        'required_delta_v_km_s': budget['required_delta_v_km_s'],
        'margin_km_s': launch['vehicle_delta_v_km_s'] - budget['required_delta_v_km_s'],
    }


def _mission_for_plan(mission, plan_delay_days):
    overrides = mission.get('plan_overrides', {}).get(plan_delay_days)
    if not overrides:
        return mission
    merged = dict(mission)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def evaluate_scenario(scenario, mission, plan_delay_days=0.0):
    launch = mission['launch']
    orbit = mission['orbit']
    spacecraft = mission['spacecraft']
    mu = orbit.get('mu_km3_s2', MU_EARTH_KM3_S2)
    total_delay_days = plan_delay_days + scenario.delay_days
    launch_tai_s = launch['nominal_tai_s'] + total_delay_days * DAY_S
    crossing_tai_s = plane_crossing_times(launch['node_longitude_rad'], launch['site_longitude_rad'], launch_tai_s)[0]

    findings = []
    inclination = launch['target_inclination_rad'] + scenario.insertion_di_rad
    try:
        ascent = _ascent(launch, orbit, inclination)
        makeup_dv_km_s = max(0.0, -ascent['margin_km_s'])
    except ValueError:
        ascent = None
        makeup_dv_km_s = 0.0
        findings.append(('fuel', 'inclination_unreachable', inclination))

    try:
        recovery_dv_km_s = insertion_recovery_delta_v_km_s(orbit['radius_km'], scenario.insertion_da_km, scenario.insertion_di_rad, mu)
    except ValueError:
        recovery_dv_km_s = float('inf')
        findings.append(('fuel', 'insertion_unrecoverable', scenario.insertion_da_km))
    try:
        extra_propellant_kg = propellant_required(spacecraft['wet_mass_kg'], (recovery_dv_km_s + makeup_dv_km_s) * 1000.0, spacecraft['isp_s'])
    except OverflowError:
        extra_propellant_kg = float('inf')
    remaining_kg = spacecraft['propellant_kg'] - extra_propellant_kg
    if remaining_kg < 0.0:
        findings.append(('fuel', 'propellant_exhausted', extra_propellant_kg))
        remaining_kg = 0.0
    fuel_check = propellant_quality.evaluate(remaining_kg, spacecraft['reserve_kg'], spacecraft['planned_burn_kg'], spacecraft.get('uncertainty_fraction', 0.0))
    findings.extend(('fuel',) + tuple(f) for f in fuel_check['findings'])

    power = mission['power']
    eclipse_margin_wh = power['eclipse_margin_wh_nominal'] - power.get('eclipse_margin_decay_wh_per_day', 0.0) * total_delay_days
    power_findings = power_quality.evaluate(power['min_soc'], power.get('max_soc', 1.0), eclipse_margin_wh, power['battery_temp_k'], power['limits'])
    findings.extend(('power',) + tuple(f) for f in power_findings)

    comm = mission['comm']
    down = set(scenario.stations_down)
    passes = [p for p in comm['passes'] if p.get('station') not in down]
    comm_findings = contact_quality.evaluate(passes, comm['minimum_daily_contacts'], comm['minimum_margin_db'], comm['maximum_outage_fraction'])
    findings.extend(('comm',) + tuple(f) for f in comm_findings)

    return {
        'scenario': scenario,
        'total_delay_days': total_delay_days,
        'launch_tai_s': launch_tai_s,
        'plane_crossing_tai_s': crossing_tai_s,
        'fuel': {
            'ascent': ascent,
            'makeup_dv_km_s': makeup_dv_km_s,
            'recovery_dv_km_s': recovery_dv_km_s,
            'extra_propellant_kg': extra_propellant_kg,
            'remaining_kg': remaining_kg,
            'check': fuel_check,
        },
        'power': {'eclipse_margin_wh': eclipse_margin_wh, 'findings': power_findings},
        'comm': {'passes_available': len(passes), 'findings': comm_findings},
        'findings': findings,
        'ready': not findings,
    }


def evaluate_plan(mission, plan_delay_days, scenarios, required_fraction=1.0):
    mission = _mission_for_plan(mission, plan_delay_days)
    results = [evaluate_scenario(s, mission, plan_delay_days) for s in scenarios]
    passed = sum(1 for r in results if r['ready'])
    fraction = passed / len(results) if results else 0.0
    try:
        nominal_margin = _ascent(mission['launch'], mission['orbit'], mission['launch']['target_inclination_rad'])['margin_km_s']
    except ValueError:
        nominal_margin = None
    return {
        'plan_delay_days': plan_delay_days,
        'scenarios_total': len(results),
        'scenarios_passed': passed,
        'pass_fraction': fraction,
        'required_fraction': required_fraction,
        'ready': fraction >= required_fraction,
        'nominal_ascent_margin_km_s': nominal_margin,
        'failures': [{'scenario': r['scenario'], 'total_delay_days': r['total_delay_days'], 'findings': r['findings']} for r in results if not r['ready']],
        'results': results,
    }


def screen_plans(mission, plan_delays_days, scenarios, required_fraction=1.0):
    plans = [evaluate_plan(mission, d, scenarios, required_fraction) for d in plan_delays_days]
    ranked = sorted(plans, key=lambda p: (
        not p['ready'],
        -p['pass_fraction'],
        -(p['nominal_ascent_margin_km_s'] if p['nominal_ascent_margin_km_s'] is not None else -math.inf),
        p['plan_delay_days'],
    ))
    recommended = ranked[0] if ranked and ranked[0]['ready'] else None
    return {'required_fraction': required_fraction, 'plans': ranked, 'recommended': recommended}
