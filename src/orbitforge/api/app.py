from __future__ import annotations
from dataclasses import asdict, is_dataclass
from fastapi import FastAPI
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.launch.robust import (
    assess_plan_robust,
    plan_from_dict,
    scenarios_from_dict,
)
from orbitforge.storage.sqlite import Store
app = FastAPI(title='OrbitForge Mission Lab', version='1.0.0')
store = Store(':memory:')

class KeplerReq(BaseModel):
    mean_anomaly: float
    eccentricity: float

class HohmannReq(BaseModel):
    r1_km: float
    r2_km: float

class LinkReq(BaseModel):
    range_km: float
    freq_hz: float

class EclipseReq(BaseModel):
    sat: list[float]
    sun: list[float]

class RotateReq(BaseModel):
    q: list[float]
    v: list[float]


class RobustLaunchReq(BaseModel):
    plan: dict
    delay_days: list[float]
    injection_errors: list[dict] | None = None
    station_cases: list[dict] | None = None
    required_fraction: float = 0.95


def _serialize(value):
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(_serialize(v) for v in value)
    return value

@app.get('/live')
def live():
    return {'status': 'live'}

@app.get('/ready')
def ready():
    return {'status': 'ready'}

@app.post('/v1/orbit/kepler')
def kepler(r: KeplerReq):
    return {'eccentric_anomaly': solve_kepler_elliptic(r.mean_anomaly, r.eccentricity)}

@app.post('/v1/maneuver/hohmann')
def h(r: HohmannReq):
    return hohmann(r.r1_km, r.r2_km)

@app.post('/v1/link/fspl')
def l(r: LinkReq):
    return {'loss_db': free_space_loss_db(r.range_km, r.freq_hz)}

@app.post('/v1/environment/eclipse')
def e(r: EclipseReq):
    return {'state': eclipse_state(Vec3(*r.sat), Vec3(*r.sun))}

@app.post('/v1/attitude/rotate')
def rotate(r: RotateReq):
    q = Quaternion(*r.q)
    v = q.rotate(Vec3(*r.v))
    return {'v': v.as_tuple()}


@app.post('/v1/launch/robust-assessment')
def robust_launch(r: RobustLaunchReq):
    """Assess one plan against the joint launch-day/injection/station scenarios.

    The plan is accepted only when at least ``required_fraction`` of scenarios
    satisfy fuel, power and communication constraints; the nominal-day optimum
    alone is never sufficient.
    """
    scenarios = scenarios_from_dict(
        r.delay_days, r.injection_errors, r.station_cases
    )
    assessment = assess_plan_robust(
        plan_from_dict(r.plan), scenarios, r.required_fraction
    )
    return _serialize(assessment)

@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
