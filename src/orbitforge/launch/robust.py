"""Scenario-based robust launch window assessment.

The geometric/energy window in :mod:`orbitforge.launch.windows` only describes
the nominal launch.  Operations do not happen on the nominal day: launches
slip, injection is never perfect and ground stations have outages.  This
module evaluates a candidate :class:`LaunchPlan` against a joint scenario set
spanning all three axes and only accepts plans that satisfy the fuel, power
and communication constraints in at least a prescribed fraction of scenarios.

All outage and contact times are offsets in seconds measured from the actual
launch of the scenario, so a station outage is experienced relative to the
mission timeline regardless of how many days the launch slipped.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from orbitforge.core.constants import (
    DAY_S,
    MU_EARTH_KM3_S2,
    OMEGA_EARTH_RAD_S,
    R_EARTH_EQUATOR_KM,
)
from orbitforge.ground.availability import combine_outages
from orbitforge.launch.ascent_budget import delta_v_budget
from orbitforge.launch.windows import launch_energy_bonus_km_s
from orbitforge.power.eclipse_budget import recharge_time_s, required_battery_wh
from orbitforge.propulsion.budget import delta_v_margin

SIDEREAL_DAY_S = 86164.0905
# Plane-crossing opportunities follow the sidereal clock while range scheduling
# follows a 24 h clock, so the geometric crossing drifts this much per day.
PLANE_CROSSING_DRIFT_S_PER_DAY = DAY_S - SIDEREAL_DAY_S
# Sun direction (and hence orbit beta angle) advances ~1 deg/day.
SUN_MEAN_MOTION_RAD_S = 2.0 * math.pi / (365.25 * DAY_S)


# ---------------------------------------------------------------------------
# Scenario definition: launch day x injection error x station availability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionError:
    """Realised injection error at spacecraft separation."""
    velocity_error_m_s: float = 0.0
    pointing_error_rad: float = 0.0


@dataclass(frozen=True)
class StationOutageCase:
    """Station availability for one scenario.

    ``outages`` maps a station name to half-open ``(start_offset_s,
    end_offset_s)`` intervals measured from the scenario launch epoch.
    """
    name: str
    outages: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchScenario:
    name: str
    delay_days: float
    injection: InjectionError
    station_case: StationOutageCase
    tags: frozenset[str] = frozenset()


NOMINAL_STATION_CASE = StationOutageCase('all_stations_available', {})


def build_scenarios(delay_days, injection_errors=None, station_cases=None):
    """Cartesian product of the three scenario axes.

    At least one value must be given on the delay axis; empty injection/outage
    axes default to a single nominal value so callers can assess launch-day
    slips alone.
    """
    delay_days = list(delay_days)
    injection_errors = list(injection_errors) if injection_errors is not None else [InjectionError()]
    station_cases = list(station_cases) if station_cases is not None else [NOMINAL_STATION_CASE]
    if not delay_days:
        raise ValueError('delay_days axis must not be empty')
    if any(d < 0.0 for d in delay_days):
        raise ValueError('launch delay cannot be negative')
    scenarios = []
    index = 0
    for delay in delay_days:
        for injection in injection_errors:
            for station_case in station_cases:
                index += 1
                tags = set()
                if delay > 0.0:
                    tags.add('slipped')
                if injection.velocity_error_m_s > 0.0 or injection.pointing_error_rad > 0.0:
                    tags.add('injection_error')
                if station_case.outages:
                    tags.add('station_outage')
                scenarios.append(LaunchScenario(
                    'scenario_%04d' % index, delay, injection, station_case, frozenset(tags)
                ))
    return scenarios


# ---------------------------------------------------------------------------
# Candidate plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedContact:
    station: str
    start_offset_s: float
    duration_s: float


@dataclass(frozen=True)
class LaunchPlan:
    name: str
    nominal_epoch_tai_s: float
    site_lat_rad: float
    azimuth_rad: float
    target_speed_km_s: float
    vehicle_delta_v_km_s: float
    wet_mass_kg: float
    propellant_kg: float
    isp_s: float
    orbit_radius_km: float
    beta0_rad: float
    battery_capacity_wh: float
    eclipse_load_w: float
    charge_net_w: float
    contacts: tuple[PlannedContact, ...] = ()
    required_contact_s: float = 0.0
    required_stations: frozenset[str] = frozenset()
    depth_of_discharge: float = 0.8
    discharge_efficiency: float = 0.95
    charge_efficiency: float = 0.95


# ---------------------------------------------------------------------------
# Scenario effects (each axis feeds at least one constraint)
# ---------------------------------------------------------------------------

def dogleg_delta_v_m_s(delay_days, target_speed_km_s):
    """Out-of-plane cleanup needed when launch is off the plane crossing.

    A slipped launch fires at the range's 24 h clock time while the inertial
    plane crossing drifts by the sidereal/solar day difference.  The site has
    rotated under the target plane, requiring a dogleg plane change.
    """
    drift_s = delay_days * PLANE_CROSSING_DRIFT_S_PER_DAY
    phase = min(abs(OMEGA_EARTH_RAD_S * drift_s), math.pi / 2.0)
    return target_speed_km_s * 1000.0 * math.sin(phase)


def injection_cleanup_delta_v_m_s(injection, target_speed_km_s):
    """Impulsive cleanup burn for an injection velocity/pointing error."""
    plane_change_m_s = target_speed_km_s * 1000.0 * math.sin(injection.pointing_error_rad)
    return math.hypot(injection.velocity_error_m_s, plane_change_m_s)


def beta_angle_rad(beta0_rad, delay_days):
    """Beta angle drifts as the Sun moves along the ecliptic (~1 deg/day)."""
    return beta0_rad + SUN_MEAN_MOTION_RAD_S * delay_days * DAY_S


def circular_orbit_period_s(radius_km):
    return 2.0 * math.pi * math.sqrt(radius_km ** 3 / MU_EARTH_KM3_S2)


def circular_eclipse_duration_s(radius_km, beta_rad):
    """Umbra duration for a circular orbit (cylindrical shadow approximation).

    Returns 0.0 when the orbit plane is tilted enough (high |beta|) that the
    trajectory never crosses the Earth's shadow cylinder.
    """
    r = radius_km
    across_track = r * math.sin(beta_rad)
    if abs(across_track) >= R_EARTH_EQUATOR_KM:
        return 0.0
    cos_beta = math.cos(beta_rad)
    if cos_beta <= 1e-12:
        return 0.0
    argument = math.sqrt(max(0.0, R_EARTH_EQUATOR_KM ** 2 - across_track ** 2)) / (r * cos_beta)
    mean_motion = math.sqrt(MU_EARTH_KM3_S2 / r ** 3)
    return 2.0 * math.asin(min(1.0, argument)) / mean_motion


def usable_contact_s(contact, station_outages):
    """Contact duration left after subtracting that station's outages."""
    start = contact.start_offset_s
    end = contact.start_offset_s + contact.duration_s
    blocked = 0.0
    for outage_start, outage_end in combine_outages(list(station_outages), []):
        blocked += max(0.0, min(end, outage_end) - max(start, outage_start))
    return max(0.0, contact.duration_s - blocked)


# ---------------------------------------------------------------------------
# Per-scenario evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: str
    passed: bool
    fuel_ok: bool
    power_ok: bool
    comm_ok: bool
    launch_epoch_tai_s: float
    required_delta_v_m_s: float
    available_delta_v_m_s: float
    delta_v_margin_m_s: float
    eclipse_s: float
    battery_required_wh: float
    battery_margin_wh: float
    recharge_s: float
    daylight_s: float
    usable_contact_s: float
    missing_stations: tuple[str, ...]
    failed_constraints: tuple[str, ...]


def evaluate_fuel(plan, scenario):
    """Two fuel gates: launcher dv capacity and spacecraft cleanup propellant.

    The launch vehicle must deliver the ascent plus dogleg plus injection
    cleanup dv; the onboard propellant (rocket equation) must also cover the
    cleanup burns alone.
    """
    rotation_bonus = launch_energy_bonus_km_s(plan.site_lat_rad, plan.azimuth_rad)
    ascent_km_s = delta_v_budget(plan.target_speed_km_s, rotation_bonus)['required_delta_v_km_s']
    cleanup_m_s = dogleg_delta_v_m_s(scenario.delay_days, plan.target_speed_km_s)
    cleanup_m_s += injection_cleanup_delta_v_m_s(scenario.injection, plan.target_speed_km_s)
    required_m_s = ascent_km_s * 1000.0 + cleanup_m_s
    vehicle_margin_m_s = plan.vehicle_delta_v_km_s * 1000.0 - required_m_s
    onboard_capable_m_s = delta_v_margin(0.0, plan.propellant_kg, plan.wet_mass_kg, plan.isp_s)
    onboard_margin_m_s = onboard_capable_m_s - cleanup_m_s
    margin_m_s = min(vehicle_margin_m_s, onboard_margin_m_s)
    available_m_s = required_m_s + margin_m_s
    return margin_m_s >= 0.0, required_m_s, available_m_s, margin_m_s


def evaluate_power(plan, scenario):
    beta = beta_angle_rad(plan.beta0_rad, scenario.delay_days)
    eclipse_s = circular_eclipse_duration_s(plan.orbit_radius_km, beta)
    battery_required_wh = required_battery_wh(
        eclipse_s,
        plan.eclipse_load_w,
        plan.depth_of_discharge,
        plan.discharge_efficiency,
    )
    period_s = circular_orbit_period_s(plan.orbit_radius_km)
    daylight_s = max(0.0, period_s - eclipse_s)
    recharge_s = recharge_time_s(
        battery_required_wh, plan.charge_net_w, plan.charge_efficiency
    )
    battery_ok = battery_required_wh <= plan.battery_capacity_wh
    recharge_ok = recharge_s <= daylight_s
    margin_wh = plan.battery_capacity_wh - battery_required_wh
    return (
        battery_ok and recharge_ok,
        eclipse_s,
        battery_required_wh,
        margin_wh,
        recharge_s,
        daylight_s,
    )


def evaluate_comm(plan, scenario):
    totals = {}
    for contact in plan.contacts:
        outages = scenario.station_case.outages.get(contact.station, ())
        totals[contact.station] = totals.get(contact.station, 0.0) + usable_contact_s(
            contact, outages
        )
    total_usable = sum(totals.values())
    missing = tuple(sorted(s for s in plan.required_stations if totals.get(s, 0.0) <= 0.0))
    ok = total_usable >= plan.required_contact_s and not missing
    return ok, total_usable, missing


def evaluate_plan(plan, scenario):
    fuel_ok, required_dv, available_dv, dv_margin = evaluate_fuel(plan, scenario)
    power_ok, eclipse_s, battery_wh, battery_margin, recharge_s, daylight_s = evaluate_power(
        plan, scenario
    )
    comm_ok, usable_s, missing = evaluate_comm(plan, scenario)
    failed = tuple(
        name for name, ok in (
            ('fuel', fuel_ok), ('power', power_ok), ('comm', comm_ok)
        ) if not ok
    )
    return ScenarioOutcome(
        scenario=scenario.name,
        passed=not failed,
        fuel_ok=fuel_ok,
        power_ok=power_ok,
        comm_ok=comm_ok,
        launch_epoch_tai_s=plan.nominal_epoch_tai_s + scenario.delay_days * DAY_S,
        required_delta_v_m_s=required_dv,
        available_delta_v_m_s=available_dv,
        delta_v_margin_m_s=dv_margin,
        eclipse_s=eclipse_s,
        battery_required_wh=battery_wh,
        battery_margin_wh=battery_margin,
        recharge_s=recharge_s,
        daylight_s=daylight_s,
        usable_contact_s=usable_s,
        missing_stations=missing,
        failed_constraints=failed,
    )


# ---------------------------------------------------------------------------
# Aggregate acceptance across the scenario set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobustAssessment:
    plan_name: str
    n_scenarios: int
    n_passed: int
    passed_fraction: float
    required_fraction: float
    accepted: bool
    failure_counts: dict[str, int]
    failing_scenarios: tuple[str, ...]
    worst_delta_v_margin_m_s: float
    worst_battery_margin_wh: float
    worst_contact_s: float
    outcomes: tuple[ScenarioOutcome, ...]


def assess_plan_robust(plan, scenarios, required_fraction=0.95):
    """Accept only if at least ``required_fraction`` of scenarios pass."""
    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError('scenario set must not be empty')
    if not 0.0 < required_fraction <= 1.0:
        raise ValueError('required_fraction must lie in (0, 1]')
    outcomes = tuple(evaluate_plan(plan, s) for s in scenarios)
    n_passed = sum(1 for o in outcomes if o.passed)
    fraction = n_passed / len(outcomes)
    failure_counts = {
        'fuel': sum(1 for o in outcomes if not o.fuel_ok),
        'power': sum(1 for o in outcomes if not o.power_ok),
        'comm': sum(1 for o in outcomes if not o.comm_ok),
    }
    return RobustAssessment(
        plan_name=plan.name,
        n_scenarios=len(outcomes),
        n_passed=n_passed,
        passed_fraction=fraction,
        required_fraction=required_fraction,
        accepted=fraction >= required_fraction,
        failure_counts=failure_counts,
        failing_scenarios=tuple(o.scenario for o in outcomes if not o.passed),
        worst_delta_v_margin_m_s=min(o.delta_v_margin_m_s for o in outcomes),
        worst_battery_margin_wh=min(o.battery_margin_wh for o in outcomes),
        worst_contact_s=min(o.usable_contact_s for o in outcomes),
        outcomes=outcomes,
    )


@dataclass(frozen=True)
class RobustSelection:
    required_fraction: float
    selected: str | None
    accepted_plans: tuple[str, ...]
    ranking: tuple[tuple[str, float, bool, float], ...]  # name, fraction, accepted, worst dv margin


def robust_select(plans, scenarios, required_fraction=0.95):
    """Rank plans; deliver the accepted plan with the most dv headroom.

    A plan that only wins on the nominal day is never returned as selected:
    acceptance is gated on the scenario pass fraction first, and ties break on
    the worst-case delta-v margin across the whole scenario set.
    """
    assessments = [assess_plan_robust(p, scenarios, required_fraction) for p in plans]
    accepted = [a for a in assessments if a.accepted]
    accepted.sort(
        key=lambda a: (a.passed_fraction, a.worst_delta_v_margin_m_s,
                       a.worst_battery_margin_wh, a.worst_contact_s),
        reverse=True,
    )
    ranking = tuple(
        (a.plan_name, a.passed_fraction, a.accepted, a.worst_delta_v_margin_m_s)
        for a in sorted(
            assessments,
            key=lambda a: (a.accepted, a.passed_fraction, a.worst_delta_v_margin_m_s),
            reverse=True,
        )
    )
    return RobustSelection(
        required_fraction=required_fraction,
        selected=accepted[0].plan_name if accepted else None,
        accepted_plans=tuple(a.plan_name for a in accepted),
        ranking=ranking,
    )


# ---------------------------------------------------------------------------
# Plain-dict construction (used by the HTTP layer)
# ---------------------------------------------------------------------------

def _station_case_from_dict(data):
    outages = {
        station: tuple(tuple(interval) for interval in intervals)
        for station, intervals in data.get('outages', {}).items()
    }
    return StationOutageCase(data.get('name', 'station_case'), outages)


def scenarios_from_dict(delay_days, injection_errors=None, station_cases=None):
    injections = [
        InjectionError(
            float(item.get('velocity_error_m_s', 0.0)),
            float(item.get('pointing_error_rad', 0.0)),
        )
        for item in (injection_errors or [{}])
    ]
    cases = [_station_case_from_dict(item) for item in (station_cases or [{'name': NOMINAL_STATION_CASE.name}])]
    return build_scenarios([float(d) for d in delay_days], injections, cases)


def plan_from_dict(data):
    contacts = tuple(
        PlannedContact(
            station=item['station'],
            start_offset_s=float(item['start_offset_s']),
            duration_s=float(item['duration_s']),
        )
        for item in data.get('contacts', [])
    )
    return LaunchPlan(
        name=data['name'],
        nominal_epoch_tai_s=float(data['nominal_epoch_tai_s']),
        site_lat_rad=float(data['site_lat_rad']),
        azimuth_rad=float(data['azimuth_rad']),
        target_speed_km_s=float(data['target_speed_km_s']),
        vehicle_delta_v_km_s=float(data['vehicle_delta_v_km_s']),
        wet_mass_kg=float(data['wet_mass_kg']),
        propellant_kg=float(data['propellant_kg']),
        isp_s=float(data['isp_s']),
        orbit_radius_km=float(data['orbit_radius_km']),
        beta0_rad=float(data.get('beta0_rad', 0.0)),
        battery_capacity_wh=float(data['battery_capacity_wh']),
        eclipse_load_w=float(data['eclipse_load_w']),
        charge_net_w=float(data['charge_net_w']),
        contacts=contacts,
        required_contact_s=float(data.get('required_contact_s', 0.0)),
        required_stations=frozenset(data.get('required_stations', ())),
        depth_of_discharge=float(data.get('depth_of_discharge', 0.8)),
        discharge_efficiency=float(data.get('discharge_efficiency', 0.95)),
        charge_efficiency=float(data.get('charge_efficiency', 0.95)),
    )
