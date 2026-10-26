# Android inference benchmark

These inference-only ports preserve the exported feature definitions and learned estimator functions. The retained model bundles are supplied for the six-model common-workload benchmark, not as a browser application.

## Runtime

The measured device was a TECNO CH6i running Android 13 (ARM64). The runtime used Termux Python 3.14.6, NumPy 2.4.4, and LightGBM 4.5.0 compiled without OpenMP. Install dependencies in your own Termux environment. Compile the included ensemble kernels with the device's C++ compiler:

```bash
c++ -O3 -std=c++17 -shared -fPIC arm_estimator_kernels.cpp -o lib_baseline_kernels.so
```

Place the native `lib_lightgbm.so` under `lightgbm-4.5.0/` next to these scripts, preserving `bundle/` and `baseline_bundles/`. Run from this directory:

```bash
python benchmark_arm_baselines.py validate --model LitePhish --count 5000
python benchmark_arm_baselines.py validate --model MUDS --count 5000
python benchmark_arm_baselines.py validate --model Roy --count 5000
python benchmark_arm_baselines.py validate --model E2Phish --count 5000
python benchmark_arm_baselines.py validate --model TabNet --count 5000
python benchmark_arm_baselines.py validate --model Ebbu --count 5000
python benchmark_arm_baselines.py suite
```

The suite qualifies all 5,000 prepared inputs and probabilities before five interleaved fresh-process trials per model at batch sizes 1 and 100. Labels must agree at threshold .5. Existing output files are preserved; use a fresh working copy for another run. The scripts use one numerical-library thread. Compare current outputs against the retained `results/phone_runtime_trials.csv`; a different device or runtime is a new hardware measurement.

The cohort is the seed-42 sample from the 93,498-row PhishTank resource copy. Raw URLs are untrusted strings and need not be opened. Roy denotes its 17-descriptor RF configuration. TabNet denotes its final stacking ensemble, not neural TabNet inference.

## Energy boundary

The public energy summaries describe 30 separate discharge-only resident-model trials (six models × five randomized blocks), each with 60 s idle, 120 s load, and 60 s idle. They use airplane mode, fixed dim screen, continuous boot-time telemetry, a pretrial cooling gate, and battery/temperature stop conditions. Sensor polarity is qualified once; never rectify each raw sample independently. Integrate signed current × voltage and divide by actual completed URLs. Idle-subtracted estimates retain negative or inconclusive values.

USB-connected telemetry is diagnostic only. Do not start a long unplugged experiment before a short continuity pilot confirms that the operating system does not suspend execution or telemetry. Device-specific sensor access and safety supervision must be established on the measured handset. The retained estimates have unknown fuel-gauge calibration uncertainty and are not externally metered energy. Loading, screen/system overhead, and idle subtraction boundaries are specified in the supplement.
