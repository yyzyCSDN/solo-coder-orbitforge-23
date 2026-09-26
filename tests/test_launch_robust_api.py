import math
from fastapi.testclient import TestClient
from orbitforge.api.app import app


def _plan(**overrides):
    plan = {
        'name': 'baseline',
        'nominal_epoch_tai_s': 1700000000.0,
        'site_lat_rad': math.radians(28.5),
        'azimuth_rad': math.radians(45.0),
        'target_speed_km_s': 7.8,
        'vehicle_delta_v_km_s': 10.0,
        'wet_mass_kg': 1000.0,
        'propellant_kg': 220.0,
        'isp_s': 300.0,
        'orbit_radius_km': 6878.0,
        'beta0_rad': 0.0,
        'battery_capacity_wh': 200.0,
        'eclipse_load_w': 100.0,
        'charge_net_w': 200.0,
        'contacts': [
            {'station': 'S1', 'start_offset_s': 3600.0, 'duration_s': 600.0},
            {'station': 'S2', 'start_offset_s': 7200.0, 'duration_s': 600.0},
        ],
        'required_contact_s': 300.0,
    }
    plan.update(overrides)
    return plan


def test_robust_assessment_endpoint_joint_scenarios():
    c = TestClient(app)
    body = {
        'plan': _plan(),
        'delay_days': [0.0, 2.0, 4.0],
        'injection_errors': [
            {'velocity_error_m_s': 0.0, 'pointing_error_rad': 0.0},
            {'velocity_error_m_s': 120.0, 'pointing_error_rad': 0.0},
        ],
        'station_cases': [
            {'name': 'all_ok', 'outages': {}},
            {'name': 's1_dark', 'outages': {'S1': [[3600.0, 4300.0]]}},
        ],
        'required_fraction': 0.9,
    }
    r = c.post('/v1/launch/robust-assessment', json=body)
    assert r.status_code == 200
    data = r.json()
    assert data['n_scenarios'] == 12
    assert data['accepted'] is True
    assert data['passed_fraction'] >= 0.9
    assert set(data['failure_counts']) == {'fuel', 'power', 'comm'}
    # outcomes carry per-axis evidence for every scenario
    assert len(data['outcomes']) == 12
    assert all(set(o['failed_constraints']) <= {'fuel', 'power', 'comm'} for o in data['outcomes'])


def test_nominal_day_optimal_plan_is_rejected_when_scenarios_fail():
    c = TestClient(app)
    body = {
        'plan': _plan(propellant_kg=25.0, battery_capacity_wh=80.0, contacts=[
            {'station': 'S1', 'start_offset_s': 3600.0, 'duration_s': 600.0},
        ]),
        'delay_days': [0.0, 3.0],
        'injection_errors': [{'velocity_error_m_s': 120.0}],
        'station_cases': [{'name': 's1_dark', 'outages': {'S1': [[3600.0, 4300.0]]}}],
        'required_fraction': 0.95,
    }
    r = c.post('/v1/launch/robust-assessment', json=body)
    assert r.status_code == 200
    data = r.json()
    assert data['accepted'] is False
    assert data['failing_scenarios']
    assert sum(data['failure_counts'].values()) > 0
