# ConflictTrace

Implementation of **ConflictTrace: Causal Relation-Guided Evidential Fusion for Asynchronous Process Fault Diagnosis**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Obtain the PRONTO multiphase-flow data through its released industrial dataset distribution. Secondary MOSEI/IEMOCAP experiments require the corresponding permitted feature/video/audio exports and official session partitions. The loader uses explicit packet arrays so that acquisition and network delivery remain separate from dataset-specific file naming.

Each packet is a NPY array. The four sources have these formats:

| Source | Array | Construction |
|---|---|---|
| `process` | float `[time,channels]` | Last 60 one-Hz process samples |
| `video` | RGB `[frames,height,width,3]`, values 0–255 | Ten uniformly selected frames from the native clip |
| `transient` | float `[features]` | Channelwise mean, standard deviation, RMS, crest factor and declared fixed-band energies |
| `logs` | integer `[tokens]` | Last 128 alarm/operation tokens; zero is padding/unknown |

Construct windows only inside released coverage. Do not infer a source outside its native file coverage. Build the token vocabulary on training runs and map unseen tokens to zero. Use the same feature ordering and band boundaries for every split.

The packet CSV has one row per native packet:

```csv
path,source,acquisition,delivery,run,split,label
raw/process_001.npy,process,60,65,run_01,train,0
```

Timestamps use seconds on a run-local clock. Acquisition is packet/window end and delivery is when it reaches the fusion node. If the dataset has no delivery clock, set delivery equal to acquisition before controlled delay injection. Labels describe packet-end state. `split` is `train`, `val` or `test`; a physical `run` cannot cross splits.

The decision CSV specifies five-second diagnostic times and state labels:

```csv
run,time,split,label
run_01,65,train,0
```

```bash
conflicttrace prepare --packets data/packets.csv --decisions data/decisions.csv --output data/clean --context 60
```

Use zero-based labels (normal = 0); preprocessing fits normalization on training packets. Complete runs stay in one split. Generate the mixed packet-stress set consumed by replay below:

```bash
conflicttrace stress --data data/clean --output data/mixed --kind mixed --seed 13
```

## Training and inference

```bash
conflicttrace train --data data/clean --output outputs/full --variant full --seed 13
```

## Evaluation

```bash
conflicttrace replay --data data/clean --model outputs/full/model.pt --output outputs/full_clean
conflicttrace replay --data data/mixed --model outputs/full/model.pt --output outputs/full_mixed --history 60
conflicttrace plot --metrics outputs/full_mixed/metrics.json --clean outputs/full_clean/metrics.json --output outputs/figures
```

Replay preserves issued decisions and appends delayed revisions within `--history` seconds.

## Comparisons and ablations

```bash
conflicttrace train --data data/clean --output outputs/mean --variant mean --seed 13
conflicttrace train --data data/clean --output outputs/sum --variant evidence_sum --seed 13
conflicttrace train --data data/clean --output outputs/no_reliability --variant no_reliability --seed 13
conflicttrace train --data data/clean --output outputs/no_conflict --variant no_conflict --seed 13
conflicttrace train --data data/clean --output outputs/no_redundancy --variant no_redundancy --seed 13
conflicttrace train --data data/clean --output outputs/no_revision --variant no_revision --seed 13
```

## Outputs

`model.pt` and `history.json` retain the selected model/losses. Replay writes `metrics.json` and `revisions.json` with packet identities, issued/revised beliefs and reliability; the plot command writes comparison figures.
