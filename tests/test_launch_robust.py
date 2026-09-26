import math
import pytest

from orbitforge.launch.robust import (
    InjectionError,
    LaunchPlan,
    PlannedContact,
    StationOutageCase,
    assess_plan_robust,
    beta_angle_rad,
    build_scenarios,
    circular_eclipse_duration_s,
    circular_orbit_period_s,
    dogleg_delta_v_m_s,
    evaluate_comm,
    evaluate_fuel,
    evaluate_plan,
    evaluate_power,
    injection_cleanup_delta_v_m_s,
    plan_from_dict,
    robust_select,
    scenarios_from_dict,
    usable_contact_s,
)

SITE_LAT = math.radians(28.5)
AZIMUTH = math.radians(45.0)
TARGET_SPEED = 7.8


def make_plan(**overrides):
    data = dict(
        name='nominal',
        nominal_epoch_tai_s=1_700_000_000.0,
        site_lat_rad=SITE_LAT,
        azimuth_rad=AZIMUTH,
        target_speed_km_s=TARGET_SPEED,
        vehicle_delta_v_km_s=10.0,
        wet_mass_kg=1000.0,
        propellant_kg=100.0,
        isp_s=300.0,
        orbit_radius_km=6878.0,
        beta0_rad=0.0,
        battery_capacity_wh=100.0,
        eclipse_load_w=100.0,
        charge_net_w=200.0,
        contacts=(PlannedContact('S1', 3600.0, 600.0),),
        required_contact_s=300.0,
        required_stations=frozenset({'S1'}),
    )
    data.update(overrides)
    return LaunchPlan(**data)


nominal_injection = InjectionError()
big_injection = InjectionError(velocity_error_m_s=120.0, pointing_error_rad=0.01)
all_ok = StationOutageCase('all_ok', {})
s1_dark = StationOutageCase('s1_dark', {'S1': ((3600.0, 4300.0),)})
dual_contacts = (
    PlannedContact('S1', 3600.0, 600.0),
    PlannedContact('S2', 7200.0, 600.0),
)


def test_scenario_grid_is_joint_cartesian_product():
    scenarios = build_scenarios([0, 1, 2], [nominal_injection, big_injection], [all_ok, s1_dark])
    assert len(scenarios) == 3 * 2 * 2
    tags = {s.name: s.tags for s in scenarios}
    assert len(tags) == len(scenarios)
    slipped = [s for s in scenarios if 'slipped' in s.tags]
    outage = [s for s in scenarios if 'station_outage' in s.tags]
    error = [s for s in scenarios if 'injection_error' in s.tags]
    assert len(slipped) == 8 and len(outage) == 6 and len(error) == 6


def test_empty_delay_axis_rejected():
    with pytest.raises(ValueError):
        build_scenarios([], [nominal_injection], [all_ok])
    with pytest.raises(ValueError):
        build_scenarios([-1.0], [nominal_injection], [all_ok])


def test_default_axes_are_nominal():
    scenarios = build_scenarios([0, 1])
    assert len(scenarios) == 2
    assert scenarios[0].injection == InjectionError()
    assert scenarios[0].station_case.outages == {}


def test_dogleg_grows_with_daily_drift():
    on_day = dogleg_delta_v_m_s(0.0, TARGET_SPEED)
    one_day = dogleg_delta_v_m_s(1.0, TARGET_SPEED)
    three_days = dogleg_delta_v_m_s(3.0, TARGET_SPEED)
    assert on_day == 0.0
    assert one_day > 100.0
    assert three_days > one_day
    # ~236 s sidereal drift in a day: omega_e * 236 s ~= 0.0172 rad
    assert abs(math.asin(one_day / (TARGET_SPEED * 1000.0)) - 7.292115e-05 * 235.9095) < 1e-6


def test_injection_cleanup_rss_of_components():
    err = InjectionError(velocity_error_m_s=30.0, pointing_error_rad=0.01)
    plane = TARGET_SPEED * 1000.0 * math.sin(0.01)
    assert abs(injection_cleanup_delta_v_m_s(err, TARGET_SPEED) - math.hypot(30.0, plane)) < 1e-9


def test_fuel_nominal_passes_but_slip_and_error_add_dv():
    plan = make_plan()
    nominal = build_scenarios([0.0], [nominal_injection], [all_ok])[0]
    slipped = build_scenarios([3.0], [big_injection], [all_ok])[0]
    ok0, req0, _, margin0 = evaluate_fuel(plan, nominal)
    ok1, req1, _, margin1 = evaluate_fuel(plan, slipped)
    assert ok0 and margin0 > 0.0
    assert req1 > req0
    assert margin1 < margin0


def test_large_injection_error_breaks_onboard_propellant():
    plan = make_plan(propellant_kg=20.0)  # ~ isp*g0*ln(1000/980) ~= 60 m/s
    scenario = build_scenarios([0.0], [InjectionError(velocity_error_m_s=150.0)], [all_ok])[0]
    ok, _, _, margin = evaluate_fuel(plan, scenario)
    assert not ok and margin < 0.0


def test_power_eclipse_at_beta_zero_and_no_eclipse_at_high_beta():
    period = circular_orbit_period_s(6878.0)
    assert 5650 < period < 5700
    eclipse0 = circular_eclipse_duration_s(6878.0, 0.0)
    assert 2130 < eclipse0 < 2160
    high_beta = math.asin(6378.137 / 6878.0)
    assert circular_eclipse_duration_s(6878.0, high_beta + 0.01) == 0.0


def test_beta_drift_with_delay_changes_power_result():
    # Start just inside the shadow-cylinder edge on the negative side; the
    # ~1 deg/day Sun drift drives the slipped launch toward longer eclipses.
    edge = math.asin(6378.137 / 6878.0)
    plan = make_plan(
        beta0_rad=-edge + 0.08,
        battery_capacity_wh=42.0,
        eclipse_load_w=100.0,
        charge_net_w=400.0,
    )
    day0 = build_scenarios([0.0])[0]
    day5 = build_scenarios([5.0])[0]
    ok0, _, _, margin0, recharge0, daylight0 = evaluate_power(plan, day0)
    ok5, eclipse5, _, margin5, recharge5, daylight5 = evaluate_power(plan, day5)
    assert ok0 and recharge0 <= daylight0
    assert not ok5
    assert eclipse5 > 0.0 and margin5 < 0.0


def test_recharge_time_gate_can_fail_independently_of_capacity():
    # Big battery passes the capacity gate but the single daylight arc is short
    # for a weak charging path.
    plan = make_plan(battery_capacity_wh=1000.0, charge_net_w=10.0)
    scenario = build_scenarios([0.0])[0]
    ok, eclipse_s, _, _, recharge_s, daylight_s = evaluate_power(plan, scenario)
    assert eclipse_s > 0 and recharge_s > daylight_s and not ok


def test_contact_subtracted_by_outage():
    contact = PlannedContact('S1', 3600.0, 600.0)
    assert usable_contact_s(contact, ()) == 600.0
    assert usable_contact_s(contact, ((3600.0, 3900.0),)) == 300.0
    assert usable_contact_s(contact, ((3500.0, 4400.0),)) == 0.0


def test_comm_outage_fails_required_station_and_volume():
    plan = make_plan()
    good = build_scenarios([0.0], [nominal_injection], [all_ok])[0]
    dark = build_scenarios([0.0], [nominal_injection], [s1_dark])[0]
    ok_good, usable_good, missing_good = evaluate_comm(plan, good)
    ok_dark, usable_dark, missing_dark = evaluate_comm(plan, dark)
    assert ok_good and usable_good == 600.0 and missing_good == ()
    assert not ok_dark and usable_dark == 0.0 and missing_dark == ('S1',)


def test_joint_scenario_passes_all_three_constraints():
    plan = make_plan()
    scenario = build_scenarios([0.0], [nominal_injection], [all_ok])[0]
    outcome = evaluate_plan(plan, scenario)
    assert outcome.passed and outcome.fuel_ok and outcome.power_ok and outcome.comm_ok
    assert outcome.launch_epoch_tai_s == plan.nominal_epoch_tai_s


def test_assess_plan_fraction_and_acceptance_gate():
    plan = make_plan()
    # 8 scenarios: nominal day always fine; slipped days with big error/outage fail
    scenarios = build_scenarios(
        [0.0, 3.0],
        [nominal_injection, big_injection],
        [all_ok, s1_dark],
    )
    assessment = assess_plan_robust(plan, scenarios, required_fraction=0.95)
    assert assessment.n_scenarios == 8
    assert assessment.passed_fraction == 0.25
    assert not assessment.accepted
    assert assessment.failing_scenarios
    assert sum(assessment.failure_counts.values()) >= 1
    # loosening the gate accepts the same plan
    loose = assess_plan_robust(plan, scenarios, required_fraction=0.25)
    assert loose.accepted


def test_all_scenarios_must_be_evaluated_nominal_only_is_rejected_as_evidence():
    plan = make_plan()
    scenarios = build_scenarios([0.0], [nominal_injection], [all_ok])
    assessment = assess_plan_robust(plan, scenarios, 1.0)
    assert assessment.accepted and assessment.passed_fraction == 1.0
    # but a plan that fails in even one non-nominal scenario cannot be delivered
    broad = build_scenarios([0.0, 5.0], [nominal_injection, big_injection], [all_ok, s1_dark])
    broad_assessment = assess_plan_robust(plan, broad, 1.0)
    assert not broad_assessment.accepted


def test_robust_select_prefers_robust_plan_over_nominal_optimum():
    # tight plan looks best nominally (low mass) but has no scenario headroom
    tight = make_plan(
        name='tight', propellant_kg=30.0, battery_capacity_wh=85.0,
        required_contact_s=300.0,
    )
    # robust plan carries dv margin for the longest slip + injection error and
    # redundant station coverage so a single outage still leaves 600 s.
    robust = make_plan(
        name='robust', propellant_kg=220.0, battery_capacity_wh=200.0,
        contacts=dual_contacts, required_contact_s=300.0,
        required_stations=frozenset(),
    )
    scenarios = build_scenarios([0.0, 2.0, 4.0], [nominal_injection, big_injection], [all_ok, s1_dark])

    tight_nominal = assess_plan_robust(tight, scenarios[:1], 1.0)
    assert tight_nominal.accepted  # nominal-only optimisation would ship it

    selection = robust_select([tight, robust], scenarios, required_fraction=0.9)
    assert selection.selected == 'robust'
    assert 'tight' not in selection.accepted_plans
    names = [row[0] for row in selection.ranking]
    assert names[0] == 'robust'


def test_no_accepted_plan_returns_none_not_nominal_winner():
    tight = make_plan(name='tight', propellant_kg=10.0, battery_capacity_wh=10.0)
    scenarios = build_scenarios([0.0, 3.0], [nominal_injection, big_injection], [all_ok, s1_dark])
    selection = robust_select([tight], scenarios, 0.95)
    assert selection.selected is None
    assert selection.accepted_plans == ()


def test_dict_builders_round_trip():
    scenarios = scenarios_from_dict(
        delay_days=[0.0, 1.0],
        injection_errors=[{'velocity_error_m_s': 50.0, 'pointing_error_rad': 0.005}],
        station_cases=[{'name': 'case_a', 'outages': {'S1': [[3600.0, 4000.0]]}}],
    )
    assert len(scenarios) == 2
    assert scenarios[0].station_case.outages['S1'] == ((3600.0, 4000.0),)
    plan = plan_from_dict({
        'name': 'from_json',
        'nominal_epoch_tai_s': 1000.0,
        'site_lat_rad': 0.5,
        'azimuth_rad': 0.8,
        'target_speed_km_s': 7.8,
        'vehicle_delta_v_km_s': 9.9,
        'wet_mass_kg': 900.0,
        'propellant_kg': 80.0,
        'isp_s': 310.0,
        'orbit_radius_km': 6900.0,
        'battery_capacity_wh': 80.0,
        'eclipse_load_w': 90.0,
        'charge_net_w': 180.0,
        'contacts': [{'station': 'S1', 'start_offset_s': 100.0, 'duration_s': 400.0}],
        'required_contact_s': 200.0,
        'required_stations': ['S1'],
    })
    assert plan.required_stations == frozenset({'S1'})
    outcome = assess_plan_robust(plan, scenarios, 1.0)
    assert outcome.n_scenarios == 2
