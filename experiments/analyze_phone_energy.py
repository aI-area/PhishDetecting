"""Archive and audit signed battery-side energy, preserving all assigned trials."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
import numpy as np
OUT=Path('your_path/phone_energy')
ADB=Path('adb')
REMOTE='/data/data/com.termux/files/home/litephish'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieve():
    archive=OUT/'phone_evidence.tar.gz'
    assert not archive.exists(), 'Do not overwrite evidence'
    command=[str(ADB),'shell','run-as','com.termux','sh',
             '/data/local/tmp/litephish_environment.sh','tar','-czf',
             REMOTE+'/energy_evidence_20261020.tar.gz','-C',REMOTE,
             'battery_results_20261020','android_battery_workload.py',
             'check_android_battery_pilot.py','android_inference_benchmark.py',
             'arm_baseline_runtime.py','lib_baseline_kernels.so',
             'lightgbm-4.5.0/lib_lightgbm.so','bundle','baseline_bundles']
    subprocess.run(command,check=True,capture_output=True)
    data=subprocess.run([str(ADB),'exec-out','run-as','com.termux','cat',
                         REMOTE+'/energy_evidence_20261020.tar.gz'],check=True,capture_output=True).stdout
    archive.write_bytes(data)
    with tarfile.open(archive) as handle:
        handle.extractall(OUT/'phone',filter='data')
    command=[str(ADB),'exec-out','cat','/data/local/tmp/run_android_energy_suite.sh']
    (OUT/'deployed_supervisor.sh').write_bytes(subprocess.run(command,check=True,capture_output=True).stdout)
    metadata=subprocess.run([str(ADB),'shell',
        'getprop ro.product.model; getprop ro.build.version.release; getprop ro.product.cpu.abi; '
        'settings get system screen_brightness; settings get system screen_brightness_mode; '
        'settings get system screen_off_timeout; ps -A -o PID,ARGS'],check=True,capture_output=True)
    (OUT/'postrun_device_state.txt').write_bytes(metadata.stdout)


def main():
    global OUT, ADB, REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('package',type=Path)
    parser.add_argument('--adb',type=Path,default=Path('adb'))
    parser.add_argument('--phone-root',default='/data/data/com.termux/files/home/litephish')
    parser.add_argument('--retrieve',action='store_true')
    args=parser.parse_args()
    import pint
    U=pint.UnitRegistry()
    OUT=args.package; ADB=args.adb; REMOTE=args.phone_root
    if args.retrieve:
        retrieve()
    assert not (OUT/'verified_analysis').exists(), 'Preserve existing analysis'
    shell=OUT/'shell'
    app=OUT/'phone/battery_results_20261020'
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert digest(OUT/'schedule.txt')==protocol['schedule_sha256']
    assert (shell/'schedule.txt').read_bytes()==(OUT/'schedule.txt').read_bytes()
    assert json.loads((shell/'protocol.json').read_text())==protocol
    assert not (shell/'failure.txt').exists() and (shell/'complete.txt').exists()
    assert (shell/'status.txt').read_text().strip()=='complete_reconnect_USB'
    for name,expected in protocol['source_sha256'].items():
        deployed=OUT/'deployed_supervisor.sh' if name.endswith('.sh') else OUT/'phone'/name
        assert digest(deployed)==expected, (name,'source mismatch')
    model_files=0
    for model in protocol['models']:
        local=ROOT/'android/bundle' if model=='LitePhish' else ROOT/'android/baseline_bundles'/model
        assert local.is_dir(), 'Missing frozen model bundle'
        deployed=OUT/'phone/bundle' if model=='LitePhish' else OUT/'phone/baseline_bundles'/model
        for p in local.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            assert digest(p)==digest(deployed/p.relative_to(local)), (model,p.name)
            model_files+=1
    qualification=json.loads((app/'pilot_sensor_check.json').read_text())
    assert qualification['status']=='PILOT_QUALIFIED'
    sign=qualification['discharge_current_sign']
    assert sign in (-1,1)
    rows=list(csv.DictReader((shell/'samples.csv').open()))
    times_s=np.array([float(r['uptime_s']) for r in rows])
    gaps=np.diff(times_s)
    assert np.isfinite(times_s).all() and (gaps>0).all() and gaps.max()<5
    assert all(r['usb_online']=='0' and r['ac_online']=='0' and r['status']=='Discharging' for r in rows)
    current=U.Quantity(np.array([float(r['current_uA']) for r in rows]),'microampere')
    voltage=U.Quantity(np.array([float(r['voltage_uV']) for r in rows]),'microvolt')
    assert np.isfinite(current.m_as('A')).all() and np.isfinite(voltage.m_as('V')).all()
    assert (np.abs(current.m_as('A'))<10).all()
    assert ((voltage.m_as('V')>2.5)&(voltage.m_as('V')<5)).all()
    nonzero=current.m_as('A')[np.abs(current.m_as('A'))>.001]
    assert len(nonzero)>.95*len(rows) and (nonzero*sign>0).mean()>.95
    power=(sign*current*voltage).to('watt')
    power_W=power.m_as('W')
    # The one pilot-derived polarity applies to every sample, without rectification.
    def energy(start,end):
        a,b=start.m_as('s'),end.m_as('s')
        assert times_s[0]<=a<b<=times_s[-1], 'No boundary extrapolation'
        t_s=np.concatenate(([a],times_s[(times_s>a)&(times_s<b)],[b]))
        p_W=np.interp(t_s,times_s,power_W)
        dt=U.Quantity(np.diff(t_s),'second')
        p=U.Quantity((p_W[:-1]+p_W[1:])/2,'watt')
        return (p*dt).sum().to('joule')
    radios=(shell/'radios.log').read_text().splitlines()
    assert radios and all('airplane=1 wifi=Wifi is disabled bluetooth=0' in line for line in radios)
    wake=(shell/'display_maintenance.log').read_text().splitlines()
    assert len(wake)>500
    temperatures=[int(r['temperature_dC'])/10 for r in rows]
    capacities=[int(r['capacity_pct']) for r in rows]
    assert max(temperatures)<43 and min(capacities)>25
    schedule=[tuple(line.split()) for line in (OUT/'schedule.txt').read_text().splitlines()]
    completed=[tuple(line.split()) for line in (shell/'completed_trials.txt').read_text().splitlines()]
    assert completed==[('qualification','LitePhish')]+schedule
    files=sorted(app.glob('block*.json'))
    files=[p for p in files if not p.stem.endswith('_events')]
    assert len(files)==30
    records=[json.loads((app/f'{trial}_{model}.json').read_text()) for trial,model in schedule]
    assert {(r['trial'],r['model']) for r in records}==set(schedule)
    per_trial=[]
    max_suspend=0
    previous_finish=None
    for record in records:
        trial,model=record['trial'],record['model']
        events=record['events']
        assert json.loads((app/f'{trial}_{model}_events.json').read_text())==events
        assert [e['phase'] for e in events]==['idle_pre_start','load_start','load_finish','idle_post_finish']
        t=[U.Quantity(e['boottime_s'],'second') for e in events]
        for a,b,expected in zip(events,events[1:],(60,120,60)):
            boot=b['boottime_s']-a['boottime_s']
            mono=b['monotonic_s']-a['monotonic_s']
            assert abs(boot-expected)<2 and abs(boot-mono)<2
            max_suspend=max(max_suspend,abs(boot-mono))
        assert record['count']>0 and record['batch_size']==1 and record['max_prediction_error']<1e-7
        if previous_finish is not None:
            assert previous_finish<=events[0]['boottime_s']
        previous_finish=events[-1]['boottime_s']
        pre=energy(t[0],t[1])/(t[1]-t[0])
        post=energy(t[2],t[3])/(t[3]-t[2])
        midpoint_pre=(t[0]+t[1])/2
        midpoint_post=(t[2]+t[3])/2
        midpoint_load=(t[1]+t[2])/2
        idle_at_load=pre+(post-pre)*((midpoint_load-midpoint_pre)/(midpoint_post-midpoint_pre))
        duration=t[2]-t[1]
        gross=energy(t[1],t[2])
        incremental=gross-idle_at_load*duration
        # Independent scalar trapezoid recomputation checks array integration.
        a,b=t[1].m_as('s'),t[2].m_as('s')
        knots=[a]+[v for v in times_s.tolist() if a<v<b]+[b]
        scalar=sum((y-x)*(float(np.interp(x,times_s,power_W))+float(np.interp(y,times_s,power_W)))/2 for x,y in zip(knots,knots[1:]))
        assert math.isclose(scalar,gross.m_as('J'),rel_tol=1e-12)
        per_trial.append(dict(trial=trial,model=model,count=record['count'],
            unique_urls_observed=min(record['count'],5000),
            duration_s=duration.m_as('s'),throughput_urls_s=record['count']/duration.m_as('s'),
            gross_energy_J=gross.m_as('J'),incremental_energy_J=incremental.m_as('J'),
            gross_mJ_per_url=(gross/record['count']).m_as('mJ'),
            incremental_mJ_per_url=(incremental/record['count']).m_as('mJ'),
            mean_load_power_W=(gross/duration).m_as('W'),
            mean_pre_idle_power_W=pre.m_as('W'),mean_post_idle_power_W=post.m_as('W'),
            interpolated_idle_power_W=idle_at_load.m_as('W'),
            prediction_max_absolute_error=record['max_prediction_error']))
    summaries=[]
    for model in protocol['models']:
        group=[r for r in per_trial if r['model']==model]
        assert len(group)==5
        summary=dict(model=model,repeats=5,total_predictions=sum(r['count'] for r in group),
                     negative_incremental_trials=int(sum(r['incremental_energy_J']<0 for r in group)))
        for field in ['gross_mJ_per_url','incremental_mJ_per_url','mean_load_power_W','throughput_urls_s']:
            vals=[r[field] for r in group]
            summary['mean_'+field]=statistics.mean(vals)
            summary['sd_'+field]=statistics.stdev(vals)
        summaries.append(summary)
    currents_raw=[int(r['current_uA']) for r in rows]
    unique=sorted(set(currents_raw))
    current_step=math.gcd(*(b-a for a,b in zip(unique,unique[1:])))
    report=dict(status='VERIFIED_COMPLETE',assigned_trials=30,verified_trials=30,
        qualification_excluded=True,models_checked_against_frozen_local_files=model_files,
        total_predictions_checked=sum(r['count'] for r in records),
        max_prediction_absolute_error=max(r['max_prediction_error'] for r in records),
        max_phase_suspend_discrepancy_s=max_suspend,
        samples=len(rows),duration_s=float(times_s[-1]-times_s[0]),
        max_sample_gap_s=float(gaps.max()),discharge_current_polarity=sign,
        negative_calibrated_power_samples=int((power_W<0).sum()),
        distinct_raw_current_values=len(unique),observed_current_step_gcd_uA=current_step,
        temperature_C_range=[min(temperatures),max(temperatures)],
        battery_percent_range=[min(capacities),max(capacities)],
        display_wake_commands=len(wake),radio_checks=len(radios),
        measurement_model='P(t)=pilot_sign*I_raw(t)[uA]*V_raw(t)[uV]*1e-12 W; E_gross=integral_load P(t)dt; E_incremental=E_gross-integral_load linearly interpolated pre/post idle power dt; mJ/URL=1000*E[J]/actual completed URLs.',
        uncertainty='Fuel-gauge calibration, lag, sensor accuracy and background OS contributions unknown. No total standard/expanded uncertainty or metrological coverage interval is estimable. SD describes five within-phone repeat variation only.',
        limitations=protocol['limitations'],display_control=protocol['display'],
        interpretation='Gross values include display, wake-command, logging and OS overhead. Idle subtraction is an estimate, not isolated CPU energy. Negative incremental values are retained. Roy is best-effort reproduction; TabNet uses descriptor-selection plus final stacking, not neural inference.',
        summary=summaries)
    output=OUT/'verified_analysis'
    output.mkdir()
    for name,values in [('per_trial_energy.csv',per_trial),('model_energy_summary.csv',summaries)]:
        with (output/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    (output/'verification.json').write_text(json.dumps(report,indent=2))
    manifest={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
