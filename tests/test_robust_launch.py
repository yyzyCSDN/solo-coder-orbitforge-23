import math
from orbitforge.launch.robust import (Scenario,build_scenarios,insertion_error_points,station_outage_sets,insertion_recovery_delta_v_km_s,evaluate_scenario,evaluate_plan,screen_plans)

def mission():
    return {
        'launch':{'site_lat_rad':math.radians(28.5),'target_inclination_rad':math.radians(51.6),'node_longitude_rad':1.0,'site_longitude_rad':0.5,'nominal_tai_s':1700000000.0,'vehicle_delta_v_km_s':11.0},
        'orbit':{'radius_km':6778.0},
        'spacecraft':{'wet_mass_kg':1000.0,'isp_s':320.0,'propellant_kg':120.0,'reserve_kg':20.0,'planned_burn_kg':40.0},
        'power':{'min_soc':0.4,'max_soc':0.95,'battery_temp_k':293.0,'eclipse_margin_wh_nominal':400.0,'eclipse_margin_decay_wh_per_day':0.0,'limits':{'min_soc':0.3,'max_soc':1.0,'min_eclipse_margin_wh':50.0,'battery_temperature_k':(273.0,313.0)}},
        'comm':{'passes':[{'station':'A','min_margin_db':6.0,'outage_fraction':0.0},{'station':'B','min_margin_db':7.0,'outage_fraction':0.1},{'station':'C','min_margin_db':8.0,'outage_fraction':0.0}],'minimum_daily_contacts':2,'minimum_margin_db':3.0,'maximum_outage_fraction':0.5},
    }

def mild_scenarios():
    return build_scenarios([0.0,1.0],insertion_error_points(5.0,math.radians(0.05)),station_outage_sets(['A','B','C'],1))

def test_error_points_cover_center_axes_corners():
    pts=insertion_error_points(5.0,0.01); assert len(pts)==9 and (0.0,0.0) in pts and (15.0,0.03) in pts and (-15.0,-0.03) in pts

def test_outage_sets_combinatorial():
    assert len(station_outage_sets(['A','B','C'],1))==4; assert len(station_outage_sets(['A','B','C'],2))==7; assert () in station_outage_sets(['A'],1)

def test_joint_scenario_product():
    assert len(mild_scenarios())==2*9*4

def test_recovery_delta_v_monotonic():
    assert insertion_recovery_delta_v_km_s(6778.0,0.0,0.0)==0.0
    assert insertion_recovery_delta_v_km_s(6778.0,0.0,0.05)>insertion_recovery_delta_v_km_s(6778.0,0.0,0.01)>0
    assert insertion_recovery_delta_v_km_s(6778.0,50.0,0.0)>0

def test_nominal_scenario_ready():
    r=evaluate_scenario(Scenario(0.0,0.0,0.0),mission()); assert r['ready'] and r['findings']==[] and r['plane_crossing_tai_s']>=r['launch_tai_s']

def test_insertion_error_stresses_fuel():
    r=evaluate_scenario(Scenario(0.0,0.0,0.05),mission()); assert not r['ready']; assert any(f[0]=='fuel' for f in r['findings'])
    assert evaluate_scenario(Scenario(0.0,0.0,0.01),mission())['ready']

def test_inclination_unreachable_flagged():
    m=mission(); m['launch']['target_inclination_rad']=math.radians(28.5)
    r=evaluate_scenario(Scenario(0.0,0.0,-0.1),m); assert not r['ready']; assert any(f[1]=='inclination_unreachable' for f in r['findings'])

def test_station_outage_stresses_comm():
    r=evaluate_scenario(Scenario(0.0,0.0,0.0,('A','B')),mission()); assert not r['ready']; assert ('comm','too_few_contacts',1) in r['findings']
    assert evaluate_scenario(Scenario(0.0,0.0,0.0,('A',)),mission())['ready']

def test_launch_delay_stresses_power():
    m=mission(); m['power']['eclipse_margin_decay_wh_per_day']=150.0
    plan=evaluate_plan(m,0.0,build_scenarios([0.0,1.0,2.0,3.0],[(0.0,0.0)],[()]),required_fraction=1.0)
    assert not plan['ready'] and plan['pass_fraction']==0.75
    assert evaluate_plan(m,0.0,build_scenarios([0.0,1.0,2.0,3.0],[(0.0,0.0)],[()]),required_fraction=0.75)['ready']
    assert all(f['findings'] and f['findings'][0][0]=='power' for f in plan['failures'])

def test_joint_stress_compounds():
    m=mission(); m['power']['eclipse_margin_decay_wh_per_day']=150.0
    scenarios=build_scenarios([0.0,1.0],insertion_error_points(5.0,0.02),station_outage_sets(['A','B','C'],1))
    plan=evaluate_plan(m,0.0,scenarios,required_fraction=1.0)
    assert 0.0<plan['pass_fraction']<1.0 and plan['scenarios_total']==len(scenarios)
    domains={f[0] for f in plan['failures'][0]['findings']}; assert domains<={'fuel','power','comm'}

def test_screen_prefers_robust_over_nominal_optimum():
    m=mission(); m['power']['eclipse_margin_decay_wh_per_day']=150.0
    m['plan_overrides']={0.0:{'launch':{'vehicle_delta_v_km_s':11.5},'power':{'eclipse_margin_wh_nominal':120.0}},1.0:{'power':{'eclipse_margin_wh_nominal':500.0}}}
    out=screen_plans(m,[0.0,1.0],build_scenarios([0.0,1.0],[(0.0,0.0)],[()]),required_fraction=1.0)
    energy_best=[p for p in out['plans'] if p['plan_delay_days']==0.0][0]
    assert energy_best['nominal_ascent_margin_km_s']>out['recommended']['nominal_ascent_margin_km_s']
    assert not energy_best['ready'] and out['recommended']['plan_delay_days']==1.0 and out['plans'][0] is out['recommended']
