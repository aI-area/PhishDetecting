"""Validate and profile frozen baseline models in fresh Android processes."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from arm_baseline_runtime import BaselineRuntime, ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['validate','trial','suite'])
    p.add_argument('--model',choices=['MUDS','Roy','E2Phish','TabNet','Ebbu','LitePhish'])
    p.add_argument('--count',type=int,default=5000)
    p.add_argument('--batch-size',type=int,choices=[1,100],default=1)
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--temperature-file',type=Path,default=Path('/sys/class/power_supply/battery/temp'),help='Battery temperature in tenths of a degree Celsius')
    args=p.parse_args()
    output=ROOT/'baseline_results'
    output.mkdir(exist_ok=True)
    if args.mode=='suite':
        for name in ['MUDS','Roy','E2Phish','TabNet','Ebbu']:
            subprocess.run([sys.executable,__file__,'validate','--model',name],check=True)
        order=[(repeat,name,batch) for repeat in range(5) for name in ['MUDS','Roy','E2Phish','TabNet','Ebbu','LitePhish'] for batch in [1,100]]
        np.random.RandomState(20261019).shuffle(order)
        schedule=output/'trial_order.json'
        assert not schedule.exists(), 'Do not replace a frozen schedule'
        schedule.write_text(json.dumps(order))
        time.sleep(120)
        for repeat,name,batch in order:
            # Delay a trial if the externally observed battery temperature is high.
            temperature=args.temperature_file
            for attempt in range(120):
                if int(temperature.read_text().strip()) <= 400:
                    break
                time.sleep(10)
            else:
                raise RuntimeError('Battery did not cool below 40 C; preserve completed trials')
            subprocess.run([sys.executable,__file__,'trial','--model',name,'--repeat',str(repeat),'--batch-size',str(batch)],check=True)
            time.sleep(10)
        (output/'suite_complete.json').write_text(json.dumps(dict(trials=len(order),cohort=5000)))
        return
    if args.model=='LitePhish':
        from android_inference_benchmark import Runtime
        runtime_class=Runtime
        reference_file=ROOT/'bundle/reference.npz'
        cohort_file=ROOT/'bundle/cohort.json'
    else:
        runtime_class=lambda:BaselineRuntime(args.model)
        reference_file=ROOT/'baseline_bundles'/args.model/'reference.npz'
        cohort_file=ROOT/'baseline_bundles'/args.model/'cohort.json'
    cohort=json.loads(cohort_file.read_text())
    urls=cohort['urls'][:args.count]
    start=time.perf_counter()
    runtime=runtime_class()
    load_s=time.perf_counter()-start
    if args.mode=='validate':
        references=np.load(reference_file)
        t0=time.perf_counter()
        raw=runtime.extract(urls)
        prepared=runtime.prepare(raw)
        probabilities=runtime.predict(prepared)
        duration=time.perf_counter()-t0
        checks={}
        for key,actual in [('raw',raw),('prepared',prepared),('probabilities',probabilities)]:
            expected=references[key][:args.count]
            checks[key]=dict(max_absolute_error=float(np.max(np.abs(actual-expected))),
                passed=bool(np.allclose(actual,expected,rtol=1e-7,atol=1e-7)))
        checks['identical_labels']=bool(np.array_equal(probabilities>=.5,references['probabilities'][:args.count]>=.5))
        record=dict(model=args.model,count=len(urls),checks=checks,load_s=load_s,duration_s=duration,
                    wall_timestamp=time.time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target=output/f'validation_{args.model}_{args.count}.json'
        assert not target.exists(), 'Preserve prior validation evidence'
        target.write_text(json.dumps(record,indent=2))
        np.savez_compressed(target.with_suffix('.npz'),probabilities=probabilities,raw=raw,prepared=prepared)
        print(json.dumps(record),flush=True)
        assert all(v['passed'] for v in checks.values() if isinstance(v,dict)) and checks['identical_labels']
        return
    if args.model!='LitePhish':
        val=json.loads((output/f'validation_{args.model}_5000.json').read_text())
        assert val['checks']['identical_labels'] and all(v['passed'] for v in val['checks'].values() if isinstance(v,dict))
    else:
        val=json.loads((ROOT/'results/validation.json').read_text())
        assert val['checks']['identical_labels']
    reference=np.load(reference_file)['probabilities'][:args.count]
    target=output/f'trial_{args.model}_{args.repeat}_batch_{args.batch_size}.json'
    assert not target.exists(), 'Do not overwrite a completed trial'
    def predict(x):
        value=runtime.predict(x)
        return value[1] if args.model=='LitePhish' else value
    predict(runtime.prepare(runtime.extract(urls[:20])))
    wall_start=time.time()
    cpu_start=time.process_time()
    start=time.perf_counter()
    stages=np.zeros((len(urls),4))
    probabilities=[]
    for i in range(0,len(urls),args.batch_size):
        part=urls[i:i+args.batch_size]
        t0=time.perf_counter()
        raw=runtime.extract(part)
        t1=time.perf_counter()
        prepared=runtime.prepare(raw)
        t2=time.perf_counter()
        prob=predict(prepared)
        t3=time.perf_counter()
        stages[i:i+len(part)]=np.asarray([t1-t0,t2-t1,t3-t2,t3-t0])/len(part)
        probabilities.extend(prob.tolist())
    elapsed=time.perf_counter()-start
    cpu=time.process_time()-cpu_start
    probabilities=np.asarray(probabilities)
    maxerror=float(np.max(np.abs(probabilities-reference)))
    assert np.allclose(probabilities,reference,atol=1e-7,rtol=1e-7),maxerror
    record=dict(model=args.model,repeat=args.repeat,batch_size=args.batch_size,count=len(urls),
        load_s=load_s,wall_start_unix=wall_start,wall_finish_unix=time.time(),duration_s=elapsed,
        mean_ms=dict(zip(['feature_extraction','preprocessing','inference','end_to_end'],(stages.mean(axis=0)*1000).tolist())),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        throughput_urls_s=len(urls)/elapsed,cpu_process_s=cpu,cpu_utilization_one_core_percent=cpu/elapsed*100,
        p50_end_to_end_ms=float(np.percentile(stages[:,3],50)*1000),
        p95_end_to_end_ms=float(np.percentile(stages[:,3],95)*1000),
        prediction_max_absolute_error=maxerror)
    target.write_text(json.dumps(record,indent=2))
    np.savez_compressed(target.with_suffix('.npz'),timings_s=stages,probabilities=probabilities)
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
