# ConflictTrace

Implementation of **ConflictTrace: Causal Relation-Guided Evidential Fusion for Asynchronous Process Fault Diagnosis**.

ConflictTrace separately controls source reliability, symmetric pair relations and fused evidence concentration. The replay engine respects both acquisition and delivery times, keeps issued actions intact, and appends revisions when delayed packets enter the retained history.

## Installation

Use Python 3.11 or later in an activated environment:

```bash
python -m pip install -e .
```

CPU is the default. Install a suitable CUDA PyTorch build and add `--device cuda` for GPU execution.

## Prepare packet and decision manifests

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

Labels are consecutive IDs starting at zero, with normal state zero. Preparation excludes decisions whose source-context packet annotations cross a state boundary. Numeric channel normalization and clipping percentiles use training packets only. Natural missingness remains in stored arrays; the encoder replaces normalized missing entries with zero while quality features retain missingness. Video quality uses exposure and Laplacian sharpness; log quality uses valid-token fraction and count. The manifest retains both clocks, run identity and observable quality.

For grouped five-fold evaluation, export three folds as train, one as validation and one as test for each rotation. Seeds and stress manifests must match across methods. No run-level split is reconstructed from shuffled windows.

## Controlled stress

```bash
conflicttrace stress --data data/clean --output data/missing50 --kind missing --rate 0.5 --seed 13
conflicttrace stress --data data/clean --output data/delay30 --kind delay --delay 30 --seed 13
conflicttrace stress --data data/clean --output data/contradiction30 --kind contradiction --rate 0.3 --seed 13
conflicttrace stress --data data/clean --output data/mixed --kind mixed --seed 13
```

`outage` removes one source for `--start` through `--start + --duration`, selected by `--source`; use durations 30, 60 and 120 seconds. `quality` clips numeric packets at training percentiles, blurs video and removes alternate log tokens. `missing` selects an exact count from existing packets. Delay changes delivery only. Mixed stress combines 30% missingness, 30-second delay, 30% degradation and 30% replacement.

Semantic replacement uses another run and class in the same split with the same source type. It chooses the donor with closest observable quality, then recomputes quality from the substituted data; this is an approximate quality-matching convention rather than an assertion that different raw packets have identical quality. Ineligible replacements remain unchanged and realized counts are recorded. Corruption labels and donor class IDs remain evaluator metadata and never enter the model.

Training augmentation uses the same command with `--split train`; derive training and evaluation stress files with different seeds. For combined training operators, pass one generated directory as the next operator's input. Validation/test statistics never refit preprocessing.

## Train

```bash
conflicttrace train --data data/clean --output outputs/full --variant full --seed 13
```

The process encoder has two temporal convolutions; video uses a frame CNN with pooling; transients use an MLP; logs use an embedding and GRU. All produce width-128 latents by default. Source heads produce softplus evidence. Absolute sigmoid gates use latent features, observable quality, log-age and disagreement with other delivered opinions.

Discounting moves each weak opinion toward vacuity. The symmetric relation head uses absolute latent differences, latent products and symmetric age features to classify complementarity, redundancy and contradiction. Pair confidence weights conflict/redundancy scores. Age-weighted discounted opinions determine class direction; conflict, redundancy and effective source count determine the separate concentration. An empty packet set returns uniform probabilities and uncertainty one.

The loss combines expected Dirichlet categorical cross-entropy, annealed wrong-class KL, Brier loss and drop-one-source arrival constraints. A small source-opinion auxiliary loss keeps individual heads discriminative. Drop-one views reuse the same encoded packets. The contradiction coefficient in the uncertainty revision term is detached; relation scores remain differentiable in fusion so the relation head can learn with no structural labels. These are explicit training conventions for the label-free relation branch.

| Option | Default | Meaning |
|---|---:|---|
| `--width`, `--batch-size` | 128, 64 | Shared latent width and accumulated decision minibatch |
| `--epochs`, `--patience` | 100, 15 | Training limit and validation stopping |
| `--lr`, `--weight-decay` | 0.0003, 0.0001 | AdamW settings |
| `--tau` | 30 | Age-weight time constant in seconds |
| `--beta`, `--kappa` | 2, 1 | Conflict suppression and multisource concentration gain |
| `--revision-weight`, `--brier-weight` | 0.1, 0.1 | Objective coefficients |
| `--revision-eta`, `--conflict-margin` | 1, 0.1 | Movement allowance and contradiction uncertainty margin |

Validation Macro-F1 selects `model.pt`, with ECE breaking differences within 0.1 percentage point. `history.json` records actual train loss and validation metrics.

## Baselines and ablations

```bash
conflicttrace train --data data/clean --output outputs/mean --variant mean --seed 13
conflicttrace train --data data/clean --output outputs/sum --variant evidence_sum --seed 13
conflicttrace train --data data/clean --output outputs/no_reliability --variant no_reliability --seed 13
conflicttrace train --data data/clean --output outputs/no_conflict --variant no_conflict --seed 13
conflicttrace train --data data/clean --output outputs/no_redundancy --variant no_redundancy --seed 13
conflicttrace train --data data/clean --output outputs/no_revision --variant no_revision --seed 13
```

Mean-probability fusion and additive evidence are fixed-fusion baselines using the same encoders. For the mean baseline, an equivalent Dirichlet concentration with minimum alpha one preserves the exact mean probability. Ablations remove absolute discounting, conflict suppression, redundancy attenuation or revision losses. Every method receives identical packet arrays, masks, clocks and optimization limits.

## Replay and metrics

```bash
conflicttrace replay --data data/clean --model outputs/full/model.pt --output outputs/full_clean
conflicttrace replay --data data/mixed --model outputs/full/model.pt --output outputs/full_mixed --history 60
```

At time `t`, an issued decision uses the most recent acquired packet per source satisfying delivery at or before `t` and acquisition inside `(t-context,t]`. Delayed packets can append revisions through `t+history`. A late old packet never replaces a newer acquired packet. Latents are cached per decision, so a source packet is encoded once even across revisions. Deliveries after the retained horizon are counted but do not rewrite history. The full-information comparator is calculated separately for regret and cannot change issued actions.

`metrics.json` includes issued-decision Macro-F1, NLL, Brier, 15-bin equal-frequency ECE, high-confidence errors at 0.9, pair conflict AUROC, quality degradation AUROC, partial-information regret, cumulative prediction variation, class flips, action changes and risk–coverage. Macro-F1 averages the full declared class taxonomy. Risk–coverage records actual retained fractions alongside requested fractions. AUROC is null when the required positive and negative examples are absent. Conflict scoring uses relation contradiction probability times discounted JSD on eligible pairs; degradation labels are excluded from that ledger.

Revision regret averages positive Brier excess over the full-information comparator across retained belief states. Variation and flips are averaged per decision. Alert identity is a non-normal class at confidence at least 0.9, so switching between fault alerts also counts as action change. `revisions.json` preserves the issued belief first, followed by every appended revision, packet IDs, relation vectors and reliability. The original action is never overwritten.

The cost output reports actual parameter count, available-packet encoder calls and issued-decision latency including packet loading and all available encoders. Offline full-information comparisons are excluded from the encoder-call count and issued latency.

```bash
conflicttrace plot --metrics outputs/full_mixed/metrics.json --clean outputs/full_clean/metrics.json --output outputs/figures
```

The plot command draws discrimination, calibration, conflict and revision comparisons and calculates clean-minus-stressed Macro-F1 for the same variant. Repeat outer run-group folds and initialization seeds before aggregating results; overlapping decisions are not independent experimental replications.
