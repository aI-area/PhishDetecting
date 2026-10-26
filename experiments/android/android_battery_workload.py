"""Battery-only online inference workload with resident-model idle controls."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from arm_baseline_runtime import BaselineRuntime, ROOT

OUT=ROOT/'battery_results_20261019'
SENSOR=Path('/data/local/tmp/litephish_energy_20261019/samples.csv')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('model')
    p.add_argument('trial')
    p.add_argument('--idle',type=int,default=60)
    p.add_argument('--load',type=int,default=120)
    p.add_argument('--output-dir',type=Path,default=OUT)
    a=p.parse_args()
    out=a.output_dir
    out.mkdir(exist_ok=True)
    path=out/f'{a.trial}_{a.model}.json'
    assert not path.exists(), 'Do not overwrite evidence'
    directory=ROOT/'bundle' if a.model=='LitePhish' else ROOT/'baseline_bundles'/a.model
    urls=json.loads((directory/'cohort.json').read_text())['urls']
    reference=np.load(directory/'reference.npz')['probabilities']
    if a.model=='LitePhish':
        from android_inference_benchmark import Runtime
        runtime=Runtime()
    else:
        runtime=BaselineRuntime(a.model)
    def predict(url):
        value=runtime.predict(runtime.prepare(runtime.extract([url])))
        return float(value[1][0]) if a.model=='LitePhish' else float(value[0])
    for i,u in enumerate(urls[:20]):
        assert abs(predict(u)-reference[i])<1e-7
    events=[]
    def phase(name):
        events.append(dict(phase=name,boottime_s=time.clock_gettime(time.CLOCK_BOOTTIME),
                           monotonic_s=time.monotonic(),wall_unix_s=time.time()))
        (out/f'{a.trial}_{a.model}_events.json').write_text(json.dumps(events))
        if len(events)>1:
            first,last=events[0],events[-1]
            suspend=(last['boottime_s']-first['boottime_s'])-(last['monotonic_s']-first['monotonic_s'])
            assert abs(suspend)<2, ('CPU suspend invalidates trial',suspend)
    phase('idle_pre_start')
    time.sleep(a.idle)
    phase('load_start')
    start=time.monotonic()
    count=0
    max_error=0.0
    while time.monotonic()-start<a.load:
        index=count%len(urls)
        value=predict(urls[index])
        error=abs(value-reference[index])
        assert error<1e-7, (index,error)
        max_error=max(max_error,error)
        count+=1
    phase('load_finish')
    time.sleep(a.idle)
    phase('idle_post_finish')
    record=dict(model=a.model,trial=a.trial,batch_size=1,count=count,unique_cohort_urls=5000,
        max_prediction_error=max_error,events=events,
        note='Model remains resident during idle controls; repeated cohort throughput is not additional independent test data.')
    path.write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
