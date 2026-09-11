# Team Handbook: Smartphone Audio Sleep-Stage Model Development

**Team:** Five researchers  
**Program:** Stage 1 - maximize teacher performance; Stage 2 - distill and deploy a lightweight student  
**Target classes:** Wake, REM, Light (N1 + N2), Deep (N3)  
**Target completion:** 27 November 2026  
**Handbook version:** 1.0 - startup-ready baseline  
**Final alignment review:** 8 September 2026  
**Companion strategy:** [MODEL_DEVELOPMENT_STRATEGY.md](./MODEL_DEVELOPMENT_STRATEGY.md)

This handbook defines how the team works. The companion strategy defines what the team is building and why.

### Document authority

When instructions conflict, use this order:

1. the signed research plan and contractual requirements;
2. `MODEL_DEVELOPMENT_STRATEGY.md` for technical scope and acceptance intent;
3. this handbook for team procedure and evidence requirements;
4. approved entries in `registry/DECISIONS.md` for project-specific choices;
5. the relevant release manifest and resolved configuration for exact execution.

A lower-level document may refine a higher-level requirement but may not weaken it silently. Any approved exception must identify the higher-level requirement, reason, owner, evidence and expiration/revisit condition.

---

# Part I - Quick Operating Guide

Part I is the mandatory first read for every researcher. It is intentionally short enough to use as a daily reference. Part II contains the detailed implementation procedures, schemas, templates, and stage-gate checklists.

## 1. Project at a Glance

### 1.1 Outcome

Build a four-class sleep-stage system from nocturnal smartphone audio:

```text
30-second smartphone audio
        |
        v
on-device acoustic front end
        |
        v
lightweight student model ------> edge-only stage probabilities
        |
        v
nightly logits / approved derived features
        |
        v
optional post-wake temporal model ------> final hybrid stage sequence
```

Raw audio remains on the phone. The hybrid path may use only approved derived outputs after the user wakes.

Current scope excludes OSA diagnosis, clinical diagnostic claims, cloud processing of raw bedroom audio, and commercial training on datasets whose rights have not been approved. Those items require a separate scope and governance decision.

### 1.2 The two stages

| Stage | Optimization target | Permitted complexity | Required outcome |
|---|---|---|---|
| **Stage 1: Teacher** | Highest reliable predictive performance and robustness | Large backbones, full-night context, expensive training, ensembles, offline inference | Frozen calibrated teacher, temporal post-processor, and versioned logits/embeddings for distillation |
| **Stage 2: Student** | Best accuracy under smartphone constraints | Mobile-friendly operations, distillation, structured pruning, QAT, duty-cycled inference | Release-ready on-device package plus optional hybrid post-processing |

Do not optimize the teacher for phone size. Do not begin broad student optimization before the teacher interface is frozen.

### 1.3 Final acceptance criteria

The final hybrid result on the locked, subject-disjoint, real-world smartphone evaluation set must meet:

| Metric / resource | Requirement |
|---|---:|
| Macro-F1 | >= 0.60 |
| Cohen's kappa | >= 0.50 |
| Accuracy | >= 0.60 |
| Full compiled on-device package | < 50 MB |
| Peak inference RAM | <= 250 MB |
| Operating test | Repeated 30-second inference across an 8-hour night |

Edge-only metrics must also be reported. The hybrid result cannot be used to hide a weak or unstable edge model.

## 2. Ten Rules Everyone Follows

1. **One frozen evaluation harness.** All comparable numbers come from `eval/`; researchers do not implement private metrics or splits.
2. **Split by participant and recording night.** Epoch-level random splitting and channel-level leakage are prohibited.
3. **Develop on validation; lock test.** The test set is touched only at formal stage gates and every access is recorded.
4. **Register before running.** Every experiment starts with a hypothesis, parent run, changed factor, success criterion, and owner.
5. **Check prior work first.** Search completed and planned experiments before spending compute.
6. **Log every outcome.** Successful, negative, failed, stopped, and invalid runs all enter the project record.
7. **Use evidence appropriate to the decision.** One seed is enough for a smoke test; claims and stage decisions require at least three seeds.
8. **Freeze the handoff.** Stage 2 uses a versioned teacher, calibration, output schema, preprocessing definition, and split manifest.
9. **Measure deployment on actual phones.** Parameter count and FLOPs are estimates; package size, RAM, latency, energy, and thermal behavior are measured.
10. **Protect privacy and reproducibility.** No raw audio enters source control, experiment logs, crash reports, or cloud transfer. Every claimed result maps to code, config, data/evaluation versions, and artifacts.

If a deadline conflicts with these rules, reduce experiment scope. Do not reduce evaluation integrity.

## 3. Daily Research Workflow

Use this loop for every experiment:

1. **Check:** search the experiment registry, recent researcher status files, and decision log.
2. **Propose:** create the experiment directory and complete `experiment.md` before training.
3. **Review scope:** confirm the work is inside the assigned lane or record an approved cross-lane collaboration.
4. **Run a smoke test:** validate data, tensor shapes, loss behavior, and evaluator compatibility on a small subset.
5. **Screen:** run the full validation experiment with the minimum compute needed to reject weak ideas.
6. **Confirm:** for a promising result, run at least three seeds and required robustness slices.
7. **Evaluate:** produce `results.json` through the shared evaluator only.
8. **Record:** append the result to the master experiment log and update the researcher's status file.
9. **Integrate or stop:** submit reusable components for review, or document why the branch is closed.

An experiment is not complete when training ends. It is complete when the result and conclusion are reproducible and visible to the team.

## 4. Team Organization by Stage

Researchers are identified as `r01` through `r05` in paths and experiment IDs. Record real names and backups in `TEAM.md`.

### 4.1 Stage 1 lanes

| Researcher | Primary lane | Required integration responsibility |
|---|---|---|
| `r01` | Data, label harmonization, evaluation harness, reproducible baselines | Maintain split/evaluation versions and reproduce candidate teachers |
| `r02` | OPERA/AudioMAE adaptation, supervised and self-supervised teacher training | Publish reusable backbone and embedding interfaces |
| `r03` | Noise, RIR/MIR, device simulation, domain robustness/adversarial training | Maintain augmentation recipes and real-domain robustness matrix |
| `r04` | Temporal context and server-side post-processing | Maintain edge-versus-hybrid evaluation and transition analysis |
| `r05` | Losses, imbalance, calibration, semi-supervision, ensemble/convergence | Assemble and ablate the final teacher candidate |

### 4.2 Stage 2 lanes

| Researcher | Primary lane | Required integration responsibility |
|---|---|---|
| `r01` | Distillation data, teacher-output cache, hard/soft target protocol | Protect split integrity and reproduce final student metrics |
| `r02` | Mobile student architecture and structured compression | Maintain MobileNetV3/EfficientNet-Lite/DS-CNN candidates |
| `r03` | Logit/feature/sequence distillation, QAT, quantization sensitivity | Produce the trained export candidate and ablations |
| `r04` | Mobile front end, runtime/export, parity, device profiling | Own package-size, RAM, latency, energy, thermal, and endurance evidence |
| `r05` | Hybrid temporal model, calibration, final integration and reporting | Assemble release candidate and final edge/hybrid evaluation |

These are primary ownership lanes, not walls. Cross-lane work is encouraged when the owner and interface are recorded. One researcher must never be the only person able to reproduce a stage-gate result.

### 4.3 Required handoffs

| Handoff | Producer | Consumer | Required artifact |
|---|---|---|---|
| Data -> all lanes | `r01` | `r02-r05` | Versioned manifests, canonical labels, feature/epoch specification |
| Backbone -> robustness/temporal | `r02` | `r03-r05` | Importable encoder, checkpoint, embedding/output contract |
| Augmentation -> teacher/student training | `r03` | `r02`, `r05`, Stage 2 | Versioned policies with clean and real-domain ablations |
| Epoch predictions -> temporal model | Teacher/student owners | `r04`/`r05` | Ordered logits, quality flags, timestamps, sequence manifest |
| Frozen teacher -> Stage 2 | Stage 1 team | Stage 2 team | Freeze manifest defined in Section 11 |
| Student -> mobile verification | `r02-r03` | `r04` | Exported model, golden fixtures, feature/output specification |

## 5. Stage Gates at a Glance

### Gate A - Foundation ready

- Canonical four-class labels and 30-second alignment are documented.
- Participant/night-disjoint splits are materialized and hashed.
- The evaluator produces Macro-F1, kappa, accuracy, per-class metrics, and edge/hybrid views.
- All five researchers pass the canonical environment/evaluator fixture; at least two independently reproduce the full reference baseline within the declared tolerance.

### Gate B - Stage 1 teacher frozen

- Best teacher and lower-cost fallback are reproduced with at least three seeds.
- Calibrated teacher plus temporal post-processor meets the project metric thresholds at the formal checkpoint.
- Teacher-only and post-processed results are both reported.
- Teacher outputs, calibration, checkpoint, preprocessing, and output schema are frozen and versioned.
- Main ablations, domain slices, failure modes, and negative experiments are documented.

### Gate C - Stage 2 release candidate

- Student architecture, distillation recipe, QAT/export path, and runtime are frozen.
- Exported/mobile outputs pass golden-fixture parity tests.
- Full compiled package is < 50 MB and peak RAM is <= 250 MB on every target device.
- Eight-hour background operation and interruption tests pass the frozen device criteria.
- Final hybrid metrics meet Macro-F1 >= 0.60, kappa >= 0.50, and accuracy >= 0.60.
- Edge-only metrics and teacher-student/compression ablations are complete.

### Gate D - Final delivery

- Model and runtime/API package are integration-ready.
- End-to-end training, distillation, export, inference, post-processing, and evaluation are reproducible.
- Performance, device, privacy, limitations, data rights, and technical reports are approved.

## Start Here - First 48 Hours

The stage owner runs one kickoff session and leaves these outputs in the repository:

1. Map the five real researchers to `r01-r05`, assign shared roles/backups, and create `TEAM.md`.
2. Create the repository skeleton from Section 6 and protect the main branch.
3. Record data availability, access owner, license status, storage location and expected delivery date for PSG-Audio, Multimodal OSA, internal PSG and any unlabeled audio.
4. Name the compute environment, artifact store and secret-management method.
5. Nominate representative lower-tier and target-tier phones for each intended operating system, even if procurement is pending.
6. Create the initial `registry/DECISIONS.md`, `EXPERIMENTS.md`, `TEST_ACCESS.md` and `RELEASES.md` files from the templates in Part II.
7. Open only the Gate A work: data cards, canonical labels, grouped split, evaluator, fixtures and baselines. Other model ideas may be registered, but substantial comparison runs wait until Gate A.

The kickoff is complete when every action has an owner and date. Unknown values are recorded as explicit open decisions; they are not filled with assumptions hidden in code.

---

# Part II - Detailed Implementation Procedures

## 6. Repository and Source-of-Truth Layout

Use this structure unless an existing implementation requires a documented variation:

```text
project/
├── README.md
├── AGENTS.md                         # Short project-wide rules for any AI/code assistant
├── TEAM.md                           # People, roles, backups, devices, stage owner
├── MODEL_DEVELOPMENT_STRATEGY.md
├── PROJECT_HANDBOOK.md
│
├── configs/
│   ├── data/
│   ├── teacher/
│   ├── student/
│   └── deployment/
│
├── data/
│   ├── README.md                     # Provenance, rights, access and retention
│   ├── manifests/                    # Versioned metadata; no raw audio
│   ├── splits/                       # Frozen participant/night split manifests
│   └── processed/                    # Generated features; normally not in Git
│
├── eval/                             # Protected shared evaluation package
│   ├── README.md
│   ├── VERSION
│   ├── labels.py
│   ├── splits.py
│   ├── metrics.py
│   ├── slices.py
│   ├── runner.py
│   ├── schemas/
│   └── fixtures/
│
├── common/                           # Reviewed reusable components
│   ├── audio/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── distillation/
│   └── deployment/
│
├── researchers/
│   ├── r01/
│   │   ├── WORKPLAN.md
│   │   ├── STATUS.md
│   │   ├── src/
│   │   └── experiments/
│   ├── r02/
│   ├── r03/
│   ├── r04/
│   └── r05/
│
├── integration/
│   ├── teacher/
│   ├── student/
│   ├── hybrid/
│   └── mobile/
│
├── registry/
│   ├── EXPERIMENTS.md                # Append-only human index
│   ├── DECISIONS.md                  # Versioned decisions and rationale
│   ├── TEST_ACCESS.md                # Every locked-set evaluation
│   └── RELEASES.md                   # Stage freeze/release manifests
│
├── reports/
│   ├── weekly/
│   ├── stage1/
│   ├── stage2/
│   └── final/
│
└── tools/                            # Logging, validation and report-generation helpers
```

### 6.1 What each source of truth owns

| Source | Owns | Must not contain |
|---|---|---|
| `data/manifests/` | Recording IDs, participant/night groups, channels, domains, label availability, checksums | Raw audio or private identifiers |
| `data/splits/` | Immutable train/validation/test/target-held-out membership | Hand-edited ad hoc exclusions |
| `eval/` | Canonical label mapping, metrics, slices, result schema and runners | Researcher-specific model logic |
| `registry/EXPERIMENTS.md` | Searchable index of every run and conclusion | Unverified handwritten metrics |
| `registry/DECISIONS.md` | Team decisions, alternatives, evidence, owner and reversal condition | General status updates |
| `researchers/*/STATUS.md` | Current evidence, interpretation, blockers, next experiments | A second master experiment log |
| `integration/` | Reviewed candidates used at stage gates and release | Exploratory one-off scripts |
| External artifact store | Checkpoints, cached logits/features, large plots, mobile builds | Secrets without access control |

### 6.2 Ownership and change control

| Area | Who may propose | Approval |
|---|---|---|
| Own researcher directory | Owner | Self for exploration; one reviewer before reuse |
| `common/` and `integration/` | Any researcher | One domain-relevant reviewer |
| `data/manifests/` metadata | Data owner | Data owner plus one reviewer |
| Existing split membership or label mapping | Any researcher with evidence | Evaluation owner plus stage lead; major version bump |
| Metric implementation or locked evaluation | Any researcher with evidence | Evaluation owner plus stage lead; no self-approval |
| Stage freeze/release | Stage owner | All five researchers sign the checklist |

### 6.3 Large artifacts

- Do not commit checkpoints, raw/processed audio, cached teacher outputs, mobile binaries, or large result archives to ordinary Git.
- Store a small manifest containing artifact URI, content hash, size, producing experiment, access class, and retention policy.
- Verify the hash after download before evaluation or distillation.
- Never rely on an unversioned path such as `latest.ckpt` in a stage-gate configuration.

## 7. Data, Labels, and Privacy Procedures

### 7.1 Dataset intake

For every dataset, create a data card recording:

- official name, version, source and acquisition date;
- permitted research and commercial uses;
- participant count, recording count, duration and population characteristics;
- audio devices/channels, sample rates and synchronization method;
- PSG scoring standard and available sleep-stage labels;
- known corruption, missingness, selection bias and clinical-population bias;
- storage location, access list, retention requirement and checksum procedure.

Do not begin training on a dataset until its provenance and use restrictions are recorded. PSG-Audio and Multimodal OSA may support research evaluation, but commercial training rights require explicit review. Internal PSG data remains research-only unless separately approved.

### 7.2 Canonical four-class mapping

The canonical labels are ordered:

```text
0 = Wake
1 = REM
2 = Light  (N1 + N2)
3 = Deep   (N3)
```

- Align all annotations to deterministic 30-second epochs.
- Map stages through a versioned table in `eval/labels.py`.
- Mark movement, unknown, unscored, and unusable partial epochs as excluded/masked according to the data specification.
- Record any source scoring-manual mismatch instead of silently normalizing it away.
- Preserve the source label and the mapped label in processed metadata.

Changing class order or mapping is a major evaluation change.

### 7.3 Leakage prevention

The grouping key is participant plus recording night. All channels and derived examples from the same grouped night remain in one split.

Prohibited leakage includes:

- random epoch splitting;
- placing microphone and tracheal channels from one night in different splits;
- fitting normalization, calibration, feature selection, or augmentation statistics on validation/test;
- self-supervised adaptation or pseudo-label training on locked sets unless a separately declared transductive protocol is approved;
- choosing exclusions after viewing test performance;
- tuning the cloud temporal model on final held-out sequences;
- caching teacher outputs without split and teacher-version metadata.

Run an automated overlap audit whenever manifests change. Gate evaluation stops if any participant, night, channel derivative, or content hash appears across split boundaries.

### 7.4 Split policy

Maintain four logical partitions:

1. **Train:** all parameter fitting, self-supervision, pseudo-labeling and augmentation-policy learning.
2. **Validation:** model selection, hyperparameters, calibration and ablation.
3. **Locked test:** accessed at Gate B and Gate C/D only.
4. **Target-domain held-out:** real smartphone/home-noise recordings reserved for final applicability evidence.

If the locked test and target-domain held-out set are the same asset, state that explicitly and treat it as a single locked set. If sample size supports it, add leave-one-dataset-out or leave-one-device-out validation without changing the locked final set.

### 7.5 Audio and derived-data privacy

- Raw audio remains in approved controlled storage during research and on the phone in deployment.
- Raw audio is never included in Git, tickets, chat messages, experiment notes, application logs or crash reports.
- Cloud transfer is disabled for raw audio.
- Treat embeddings as potentially identifying, not automatically anonymous.
- Prefer logits plus quality flags when they provide sufficient hybrid performance.
- Minimize embedding dimension, encrypt transfer/storage, restrict access and define deletion/retention.
- Use non-identifying recording IDs in every artifact.
- Complete a privacy review before any real-user unlabeled audio enters self-supervised learning or distillation.

## 8. Shared Evaluation Harness

### 8.1 Required evaluator behavior

`eval/runner.py` must:

- load an immutable split manifest;
- validate model output order and epoch alignment;
- compute all metrics from stored predictions and labels;
- reject duplicate/missing epochs unless an approved missing-data rule applies;
- produce separate epoch/edge and hybrid/temporal result blocks;
- produce dataset, device/channel, noise/SNR and available demographic/clinical slices;
- store the evaluation, split, preprocessing, calibration and model versions;
- emit a schema-validated `results.json` without manual metric entry.

### 8.2 Mandatory metrics

All formal runs report:

- Macro-F1 as the primary metric;
- Cohen's kappa and accuracy as mandatory secondary metrics;
- per-class precision, recall, F1 and support;
- confusion matrix;
- expected calibration error or an agreed calibration metric;
- transition-violation and stage-duration plausibility for temporal outputs;
- mean and standard deviation for confirmatory multi-seed runs;
- subject-level bootstrap confidence intervals at stage gates.

Teacher comparisons additionally report domain robustness and calibration. Student comparisons additionally report compiled package size, peak RAM, feature/inference latency, energy/thermal observations and endurance status.

### 8.3 Example result schema

```json
{
  "experiment_id": "s1-r03-e014",
  "stage": 1,
  "status": "CONFIRMATORY",
  "owner": "r03",
  "parent_experiment": "s1-r02-e008",
  "git_commit": "<commit>",
  "config_hash": "sha256:<hash>",
  "data_version": "data-1.0.0",
  "split_version": "split-1.0.0",
  "preprocess_version": "prep-1.1.0",
  "eval_version": "eval-1.0.0",
  "teacher_version": null,
  "seeds": [17, 42, 91],
  "metrics": {
    "edge": {
      "macro_f1_mean": 0.0,
      "macro_f1_std": 0.0,
      "kappa_mean": 0.0,
      "accuracy_mean": 0.0,
      "per_class_f1": {
        "wake": 0.0,
        "rem": 0.0,
        "light": 0.0,
        "deep": 0.0
      }
    },
    "hybrid": null
  },
  "efficiency": {
    "weights_mb": null,
    "compiled_package_mb": null,
    "peak_ram_mb": null,
    "latency_p50_ms": null,
    "latency_p95_ms": null
  },
  "artifacts": {
    "predictions_manifest": "<uri-or-path>",
    "checkpoint_manifest": "<uri-or-path>"
  }
}
```

Zeros above are schema placeholders, never example performance claims.

### 8.4 Version rules

- **Patch:** bug fix that provably does not change existing results.
- **Minor:** additional metric or slice; existing primary results remain valid.
- **Major:** split, label, preprocessing or metric change that can change existing results.

For a major version change:

1. record the reason and evidence in `registry/DECISIONS.md`;
2. mark affected historical results as stale, never delete them;
3. rerun the current baseline and top three active candidates;
4. use only the new version for later comparisons.

### 8.5 Statistical decision policy

- **Smoke/feasibility:** one seed; no performance claim.
- **Screening:** one seed or a reduced budget; may reject an idea but cannot establish a winner.
- **Confirmatory:** at least three fixed seeds; required before integration.
- **Stage gate:** at least three seeds plus subject-level confidence intervals and locked-set evaluation.

Before the first confirmatory comparison, record the minimum meaningful Macro-F1 difference in `registry/DECISIONS.md`. Base it on observed seed and subject uncertainty. When two models are practically tied, prefer in this order:

1. stronger worst-domain and per-class behavior;
2. simpler and more reproducible training in Stage 1;
3. smaller package, lower RAM, faster device latency and lower energy in Stage 2;
4. simpler operational/privacy boundary.

Do not describe a sub-threshold delta as an improvement.

## 9. Experiment Lifecycle and Records

### 9.1 Experiment identity

Use:

```text
s<stage>-r<researcher>-e<sequence>-<short-slug>
```

Examples:

```text
s1-r02-e008-audiomae-full-ft
s1-r03-e014-rir-mir-dynamic-snr
s2-r02-e006-mobilenetv3-w075
s2-r03-e011-logit-feature-qat
```

The ID is immutable and appears in the directory name, config, results, artifact manifest, experiment log and commit/PR title.

### 9.2 Experiment directory

```text
researchers/rXX/experiments/<experiment-id>/
├── experiment.md          # Written before the run
├── config.yaml            # Complete resolved configuration
├── environment.json       # Software and hardware identity
├── results.json           # Written by eval/runner.py only
├── notes.md               # Observations, failures and interpretation
└── artifacts.json         # Hashes and locations; not large files themselves
```

### 9.3 Required pre-run proposal

`experiment.md` must answer:

```markdown
# <experiment-id>

**Owner:** rXX
**Stage:** 1 or 2
**Status:** PROPOSED
**Parent:** <experiment-id or none>
**Tags:** <approved taxonomy>

## Hypothesis
One falsifiable sentence.

## Single major change
What changes from the parent, and what remains fixed?

## Prior-work check
- Related completed runs:
- Related planned/running work:
- Why this is not a duplicate:

## Decision metric
Primary metric/slice/resource and minimum effect needed to continue.

## Compute level
SMOKE / SCREENING / CONFIRMATORY / GATE

## Risks
Leakage, licensing, privacy, compute or integration risks.
```

Register the planned experiment in the master log before using substantial compute. This prevents two researchers from unknowingly running the same idea.

### 9.4 Status values

| Status | Meaning |
|---|---|
| `PROPOSED` | Defined and checked for duplication; not started |
| `RUNNING` | Compute in progress |
| `COMPLETE` | Evaluation and conclusion recorded |
| `FAILED` | Technical failure; cause and salvageable evidence recorded |
| `STOPPED` | Intentionally terminated by a predefined rule |
| `INVALID` | Result cannot support a claim because of leakage, evaluator mismatch or corrupted data |
| `SUPERSEDED` | Replaced by a later run; remains in history |

### 9.5 Model-specific tag taxonomy

Use stable prefixes; add new values through `registry/DECISIONS.md`.

| Prefix | Examples |
|---|---|
| `DATA-*` | `DATA-psgaudio`, `DATA-multimodal-osa`, `DATA-unlabeled` |
| `FRONT-*` | `FRONT-logmel`, `FRONT-pcen`, `FRONT-multires`, `FRONT-context` |
| `AUG-*` | `AUG-specaugment`, `AUG-rir`, `AUG-mir`, `AUG-noise-snr`, `AUG-device` |
| `TEACH-*` | `TEACH-opera`, `TEACH-audiomae`, `TEACH-cnn` |
| `TEMP-*` | `TEMP-bilstm`, `TEMP-tcn`, `TEMP-transformer`, `TEMP-crf` |
| `LOSS-*` | `LOSS-ce`, `LOSS-focal`, `LOSS-supcon`, `LOSS-domainadv`, `LOSS-boundary` |
| `SSL-*` | `SSL-contrastive`, `SSL-masked`, `SSL-pseudolabel`, `SSL-consistency` |
| `ENS-*` | `ENS-average`, `ENS-stack`, `ENS-soup` |
| `STUDENT-*` | `STUDENT-mobilenetv3`, `STUDENT-efficientnetlite`, `STUDENT-dscnn` |
| `KD-*` | `KD-logit`, `KD-feature`, `KD-sequence`, `KD-ensemble` |
| `COMP-*` | `COMP-structured-prune`, `COMP-qat`, `COMP-int8`, `COMP-mixed` |
| `DEPLOY-*` | `DEPLOY-ios`, `DEPLOY-android`, `DEPLOY-dutycycle`, `DEPLOY-parity` |
| `SYSTEM-*` | `SYSTEM-edge`, `SYSTEM-hybrid`, `SYSTEM-logits-only`, `SYSTEM-embedding` |

Do not encode every hyperparameter as a tag. Hyperparameters belong in the config; tags identify search space and prevent duplication.

### 9.6 Master experiment log

`registry/EXPERIMENTS.md` is append-only. Corrections are new entries referencing the old row.

```markdown
| ID | Status | Date | Owner | Tags | Parent | Eval | Macro-F1 | Kappa | Accuracy | Seeds | Resource gate | Conclusion |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|
| s1-r02-e008-audiomae-full-ft | COMPLETE | YYYY-MM-DD | r02 | TEACH-audiomae | s1-r01-e001-baseline | eval-1.0.0 | <from results> | <from results> | <from results> | 3 | N/A | <one evidence-based sentence> |
```

Never type or copy metric values from a notebook. Populate them from validated `results.json`, ideally through a helper tool.

### 9.7 When to stop or promote an experiment

Stop when:

- the smoke test fails the technical feasibility criterion;
- validation is clearly below the parent after the predefined budget;
- the method violates data, privacy, licensing or deployment constraints;
- a duplicate run already answers the hypothesis;
- the expected gain cannot justify the added system cost.

Promote to confirmatory when:

- the screening result exceeds the meaningful-difference threshold or resolves a critical risk;
- per-class and worst-domain results do not show unacceptable regressions;
- the code and configuration are reproducible;
- the method can be integrated through a stable interface.

## 10. Stage 1 Execution Procedure

### 10.1 Stage 1 objective and selection rule

Stage 1 maximizes validation Macro-F1 subject to reliable kappa, accuracy, calibration and real-domain robustness. Teacher size, latency and RAM are not selection constraints. A stronger calibrated ensemble may be selected over a single model because Stage 1 inference is offline.

The teacher must produce an edge-compatible per-epoch output in addition to any full-night temporal output. Stage 2 needs the former for logit/feature distillation; the latter establishes the hybrid upper bound.

### 10.2 Step 1 - Freeze the foundation

`r01` coordinates:

1. dataset cards and permissions;
2. canonical four-class mapping and 30-second alignment;
3. group-disjoint split manifest and overlap audit;
4. canonical evaluator and result schema;
5. simple log-Mel CNN, pretrained-encoder probe and temporal baseline;
6. the canonical environment/evaluator fixture run by all researchers and independent full-baseline reproduction by at least two researchers.

No broad teacher search starts until Gate A passes. Small feasibility spikes may run, but their metrics cannot enter the comparable leaderboard.

### 10.3 Step 2 - Parallel broad exploration

Run distinct, coordinated lanes:

- `r02`: OPERA and AudioMAE frozen probe, partial fine-tuning, full fine-tuning and sleep-domain self-supervised adaptation;
- `r03`: SpecAugment, RIR, MIR, dynamic SNR, device/channel response, gain/clipping/codec and realistic corruption curricula;
- `r04`: neighboring-epoch context, Bi-LSTM, TCN, Transformer/Conformer and transition-aware post-processing;
- `r05`: class balance, focal/boundary/supervised-contrastive objectives, calibration, pseudo-labeling and ensemble methods;
- `r01`: data/evaluator reliability, baseline variants, cross-dataset slices and reproduction.

Each lane first isolates factors against a common parent. Combination experiments occur only after individual effects are understood.

OPERA, AudioMAE and the named temporal/model families are required starting candidates, not an exclusive list. A researcher may propose another method when the hypothesis, compatibility, compute budget and decision criterion are registered before the run.

### 10.4 Step 3 - Focused deepening

At the midpoint review:

1. rank evidence by confirmatory validation Macro-F1;
2. examine REM and Deep performance, worst-domain behavior and calibration;
3. close unsupported branches explicitly;
4. select two or three promising teacher families/components;
5. assign at least two researchers to independently reproduce or extend each critical direction;
6. reserve compute for ablation and multi-seed confirmation.

This is deliberate overlap. Early lanes maximize coverage; focused deepening uses independent replication to reduce the risk of selecting a lucky or owner-specific implementation.

### 10.5 Step 4 - Combine and ablate components

Move proven components into `common/` or `integration/teacher/`. Construct candidates from reusable elements:

- acoustic front end;
- pretrained backbone;
- augmentation recipe;
- supervised and auxiliary losses;
- temporal head/post-processor;
- calibration method;
- ensemble rule.

For every combined candidate, run a component ablation. A component remains only if it provides a repeatable accuracy, robustness, calibration or operating advantage.

### 10.6 Step 5 - Select and calibrate the teacher

Select:

- the strongest calibrated teacher or ensemble for distillation;
- a strong single-model fallback;
- the Stage 1 server temporal post-processor;
- one edge-compatible feature/output interface.

Calibrate on validation only. Cache logits only after calibration and final preprocessing are frozen. Measure ensemble disagreement, per-class calibration and teacher confidence under severe noise before using confidence-based distillation weights.

### 10.7 Stage 1 outputs

Store a Stage 1 release manifest containing:

- checkpoint(s) and content hashes;
- full resolved training/inference config;
- code commit and environment;
- data, split, preprocessing, evaluation and calibration versions;
- class order, epoch/timestamp rules and output schema;
- selected embedding layer(s), dimensions and projection requirements;
- edge and hybrid validation/test results;
- dataset/device/noise/per-class results;
- known failure modes and valid-use limitations;
- deterministic teacher-target generation command or job definition.

## 11. Teacher-to-Student Handoff

The handoff is a controlled release, not a file copy.

### 11.1 Frozen handoff package

The package must include:

```text
teacher-version/
├── RELEASE.md
├── model-manifest.json
├── preprocessing.yaml
├── calibration.json
├── output-schema.json
├── embedding-schema.json
├── checkpoint-manifest.json
├── reference-results.json
├── golden-input-manifest.json
├── golden-output.npz
└── generate-targets.yaml
```

### 11.2 Teacher-output cache requirements

Every cached record contains:

- non-identifying recording ID, epoch index and timestamp;
- split and source dataset;
- calibrated four-class logits/probabilities in canonical order;
- selected embedding, if approved;
- teacher confidence and input-quality values;
- teacher, calibration, preprocessing and cache-schema versions;
- checksum and generation status.

Generate train outputs only for distillation training. Generate validation outputs only for analysis and loss tuning without fitting student parameters on them. Never use locked test teacher outputs for student training.

### 11.3 Change after freeze

A proposed teacher update after the handoff must state:

- why the expected gain justifies regeneration and rerunning student comparisons;
- which cached targets and results become stale;
- migration and rollback plan;
- approval from the Stage 1 owner, Stage 2 owner and evaluation owner.

Silent teacher replacement is prohibited.

## 12. Stage 2 Execution Procedure

### 12.1 Step 1 - Establish the deployment baseline

Before extensive distillation:

1. choose representative lower-tier and target-tier iOS/Android devices;
2. select candidate mobile runtime/export paths through a small feasibility spike;
3. implement the canonical log-Mel/PCEN front end and golden parity fixtures;
4. compile a simple MobileNetV3/EfficientNet-Lite/DS-CNN baseline;
5. measure full package size, peak RAM, feature time, inference latency and startup behavior;
6. document unsupported or slow operators.

This turns the model-size requirement into a real package budget and prevents late discovery that a theoretically efficient architecture performs poorly on the selected runtime.

### 12.2 Step 2 - Select the student family

Compare mobile-friendly dense models under the same feature input and training baseline:

- MobileNetV3-Small/Large variants;
- EfficientNet-Lite variants;
- compact depthwise-separable CNN;
- optional small causal TCN/GRU only when its measured gain justifies the added state and energy.

Search width, depth, expansion ratio, input resolution, Mel bins and embedding dimension. Use a Pareto view of Macro-F1, worst-class F1, package size, RAM and on-device latency. Do not choose by FLOPs alone.

### 12.3 Step 3 - Distillation ladder

Test one increment at a time:

1. hard PSG labels only;
2. calibrated teacher-logit distillation;
3. logit plus feature distillation;
4. robustness consistency across clean/noisy/device views;
5. optional sequence/transition distillation;
6. ensemble-teacher distillation;
7. permitted unlabeled-data distillation with confidence and quality filtering.

The default loss family is:

```text
L = alpha * L_hard
  + beta  * T^2 * KL(teacher_T, student_T)
  + gamma * L_feature
  + delta * L_sequence
  + eta   * L_robustness
```

Tune coefficients and temperature on validation. Use per-class balancing so Light does not dominate. Down-weight or exclude low-confidence teacher predictions, especially near stage boundaries and under severe corruption. On unlabeled data, omit the hard-label term.

### 12.4 Step 4 - Compression and quantization

Use this order:

1. train/distill the dense floating-point student;
2. apply structured channel pruning only if it improves actual device behavior;
3. fine-tune after pruning;
4. remove, replace or fuse poorly supported operators;
5. perform integer quantization-aware training using representative domains and noise;
6. use selective mixed precision only for verified sensitive layers;
7. export, compile and rerun accuracy/parity checks;
8. benchmark again on every target device.

Unstructured sparsity is not a default because it may reduce nonzero weights without reducing mobile latency or memory.

### 12.5 Step 5 - Mobile front end and duty cycling

Freeze a feature specification covering:

- sample rate and resampling;
- 30-second framing and timestamp alignment;
- window, FFT, hop, Mel filter bank and frequency limits;
- log/PCEN transform, normalization, clipping and padding;
- interrupted/missing audio and clock-drift behavior;
- output data type, shape and tolerance.

Use native low-power capture/DSP components appropriate to the production platform. Buffer one epoch, run feature extraction and inference in a short burst, store approved derived outputs, release temporary memory and return to low-power operation.

Golden tests must compare the training/reference front end with the mobile implementation on clean, noisy, clipped, short, interrupted and resampled fixtures.

### 12.6 Step 6 - Hybrid integration

The hybrid post-processor consumes the nightly student-logit sequence and optional approved quality/embedding fields. It must:

- align timestamps and handle missing epochs;
- preserve the edge result for audit;
- improve temporal plausibility without erasing real awakenings or short REM transitions;
- be evaluated in logits-only and logits-plus-embedding modes;
- remain versioned independently from the edge model.

Prefer logits-only unless embeddings provide a material, reproducible gain that justifies their privacy and operational cost.

### 12.7 Step 7 - Device and endurance verification

| Test | Required evidence |
|---|---|
| Functional parity | Mobile versus reference logits on golden fixtures within frozen tolerance |
| Package size | Full compiled application inference component < 50 MB |
| Peak RAM | Cold-start and steady-state peak <= 250 MB |
| Latency | Median and p95 for feature extraction and model inference |
| Eight-hour endurance | Completion, crash/kill count, battery change and thermal state |
| Background interruptions | Suspend/resume, calls, alarms, route changes, low battery/storage |
| Acoustic robustness | Distance, orientation, bedding, fan/HVAC, traffic, speech/TV, room/device variation |
| Privacy | No raw-audio transfer or accidental payload in telemetry/crash reports |

Latency, battery and thermal limits are not specified in the research plan. Establish baseline measurements during the deployment spike, approve explicit release thresholds in `registry/DECISIONS.md`, and freeze them before final tuning.

### 12.8 Student selection rule

A student is eligible for release only if it meets every hard resource and reliability gate. Among eligible students:

1. select by validation Macro-F1;
2. reject disproportionate REM/Deep or worst-domain collapse;
3. prefer the smaller/faster/lower-energy model when performance is practically tied;
4. verify the chosen exported artifact, not only the training checkpoint.

The planning target is to keep student Macro-F1 within three percentage points of the frozen teacher. This is an internal guardrail, not a contract requirement, and may be revised through a recorded decision.

## 13. Collaboration, Review, and Version Control

### 13.1 Shared roles

In addition to research lanes, assign these roles in `TEAM.md`:

| Role | Responsibility | Backup required? |
|---|---|---|
| Stage owner | Scope, schedule, convergence and gate readiness | Yes |
| Data steward | Rights, manifests, labels, access and privacy | Yes |
| Evaluation steward | Split/metric integrity, locked-set access and result validation | Yes |
| Integration owner | Moves reviewed components into integration branches | Yes |
| Artifact custodian | Checkpoints, hashes, cache/build manifests and retention | Yes |
| Mobile release owner | Runtime builds, device matrix and final package | Yes, Stage 2 |

One person may hold multiple roles, but no person approves their own protected change alone.

Create `TEAM.md` at kickoff:

```markdown
# Team and Project Runtime

**Current stage:** Foundation / Stage 1 / Stage 2 / Finalization
**Stage owner:**
**Target completion:** 2026-11-27

## Researchers
| ID | Name | Primary lane | Shared role | Backup | Availability |
|---|---|---|---|---|---|
| r01 | | | | | |
| r02 | | | | | |
| r03 | | | | | |
| r04 | | | | | |
| r05 | | | | | |

## Project infrastructure
- Source repository / main branch:
- Compute environments:
- Artifact store:
- Data storage and access owner:
- Secret-management method:
- Communication channel:

## Target deployment matrix
| OS | Device / chipset | Tier | OS version | Owner | Available date |
|---|---|---|---|---|---|

## Immediate open decisions
| Decision | Owner | Due date | Blocking work | Record ID |
|---|---|---|---|---|
```

### 13.2 Branching and review

| Item | Convention |
|---|---|
| Main branch | Protected; always reproducible |
| Research branch | `rXX/<stage>-<topic>`, for example `r03/s1-device-augmentation` |
| Commit | Include experiment ID when the commit supports a run |
| Pull request | State hypothesis, parent, evidence, compatibility and rollback |
| Shared code | One domain-relevant reviewer |
| Evaluation/data changes | Evaluation/data steward plus stage lead |
| Release | All five researchers sign the stage checklist |

Researchers may iterate freely in their own directory. Code becomes project-supported only after it moves through review into `common/` or `integration/`.

### 13.3 Integration requirements

A component is ready to integrate when it has:

- an importable interface rather than notebook-only or monolithic code;
- unit tests for shapes, dtypes and key edge cases;
- a complete config and environment record;
- at least one confirmatory experiment;
- evidence against the shared parent;
- no known split, license or privacy violation;
- documentation of required inputs, outputs and expected resource cost.

### 13.4 Conflict and contradiction procedure

When researchers obtain conflicting results:

1. do not average or vote on conclusions;
2. compare data, split, preprocessing, parent, seed, augmentation order and evaluator versions;
3. swap configs or reproduce each other's run;
4. identify the smallest controlled experiment that separates the hypotheses;
5. record the resolution or remaining uncertainty in `registry/DECISIONS.md`.

Contradictions are high-value findings because they often reveal hidden implementation or domain interactions.

## 14. Communication and Meeting Rhythm

The process is platform-neutral. Use the team's chosen chat/project tool, but the repository remains the source of truth.

| Cadence | Activity | Owner |
|---|---|---|
| Before each substantial run | Prior-work check and experiment registration | Experiment owner |
| After each completed/failed run | Results validation and log update | Experiment owner |
| At least twice weekly | Update `researchers/rXX/STATUS.md` | Each researcher |
| Midweek, 15 minutes | Blockers, duplicate work, compute/device conflicts | Stage owner + affected researchers |
| Weekly, 30 minutes | Evidence synthesis, contradictions, lane changes, next budget | All five; rotating digest owner |
| Stage midpoint | Broad-to-focused convergence decision | All five |
| Stage gate | Checklist, locked evaluation, release manifest | All five |

Meetings are for decisions and blockers, not reading status aloud. The weekly digest is a pre-read.

### 14.1 Weekly digest

Write `reports/weekly/<ISO-week>.md` with:

- one-sentence headline;
- current teacher or student leaderboard using comparable evaluation versions;
- per-researcher focus and strongest evidence;
- confirmed gains and dead ends;
- contradictions requiring resolution;
- planned work collisions;
- data/evaluation/privacy/device risks;
- next experiments and compute/device allocation;
- approaching stage-gate gaps.

If a chat summary is posted, link to the repository digest and keep it brief. Communication automation is optional and must not become a second source of truth.

### 14.2 Execution calendar

| Period | Stage | Team focus | Required milestone |
|---|---|---|---|
| June-July 2026 | Stage 1 | Data intake, labels, grouped splits, evaluation, baselines | Gate A |
| July-August 2026 | Stage 1 | Parallel teacher/front-end/robustness/temporal exploration | Comparable candidate leaderboard |
| August-September 2026 | Stage 1 | Focused deepening, combination, ablation, calibration | Gate B teacher freeze |
| Early October 2026 | Stage 2 | Runtime/device spike and student-family selection | Deployment baseline |
| October-early November 2026 | Stage 2 | Distillation ladder, compression, QAT and profiling | Gate C candidate |
| November-27 November 2026 | Stage 2 | Hybrid integration, locked evaluation, endurance and documentation | Gate D final release |

If work starts later, preserve the order and gate requirements. Replan scope and compute rather than deleting the test lock, teacher freeze, parity checks or device-endurance period.

### 14.3 Researcher status template

`researchers/rXX/STATUS.md`:

```markdown
# rXX Status

**Updated:** YYYY-MM-DD
**Stage / lane:**
**Current parent:**

## Current conclusion
What the evidence now supports, in 2-4 bullets.

## New evidence
| Experiment | Result | Comparator | Verdict |
|---|---|---|---|

## Building on / conflicting with others
- Related work:
- Reproduction or contradiction:
- Planned coordination:

## Blockers and risks
- Data/evaluation:
- Compute/artifact:
- Integration/device/privacy:

## Next experiments
Include IDs, hypotheses, tags and decision criteria.
```

## 15. Decision and Release Records

### 15.1 Decision record template

Append to `registry/DECISIONS.md`:

```markdown
## DEC-YYYY-NNN: <decision>

**Date:**
**Owner:**
**Status:** PROPOSED / APPROVED / REVERSED
**Applies from:**

### Context
What decision is required and why now?

### Options considered
1. Option and trade-off
2. Option and trade-off

### Evidence
Experiment IDs, reports and measurements.

### Decision
Chosen option and scope.

### Consequences
What becomes required, stale or out of scope?

### Revisit trigger
What new evidence would reopen this decision?
```

Decisions that must be recorded include split/metric changes, meaningful-difference threshold, teacher freeze, student/runtime choice, mobile latency/energy/thermal gates, embedding transfer, dataset-use interpretation and stage release.

### 15.2 Locked test access

Every access adds a row to `registry/TEST_ACCESS.md`:

```markdown
| Date | Gate | Model/release | Eval version | Split version | Commit | Requester | Approver | Reason | Result path |
```

Exploratory test access is not allowed. If a bug invalidates a locked evaluation, record the invalidation and corrective action; do not erase the access.

### 15.3 Release manifest

Each Stage 1 freeze and Stage 2 release records:

- release ID and date;
- responsible people and approvals;
- code commit and build environment;
- data/split/preprocessing/evaluation versions;
- teacher/student/post-processor/calibration versions;
- artifact locations, hashes and sizes;
- reference and locked-set result paths;
- device-build and benchmark paths when applicable;
- privacy/license review status;
- known limitations, rollback target and compatibility statement.

## 16. Stage-Gate Checklists

### 16.1 Gate A - Foundation ready

- [ ] Dataset cards and permission statuses exist.
- [ ] Canonical label mapping and class order are tested.
- [ ] Participant/night/channel overlap audit passes.
- [ ] Split manifests and preprocessing are versioned.
- [ ] Evaluator computes required overall, per-class and slice metrics.
- [ ] Edge and hybrid output schemas are separated.
- [ ] All researchers pass the canonical fixture; at least two independently reproduce the full baseline within the agreed tolerance.
- [ ] Meaningful-difference policy and fixed confirmatory seeds are recorded.
- [ ] Stage 1 lanes, backup owners and initial experiment budget are assigned.

### 16.2 Gate B - Teacher frozen

- [ ] Best teacher and single-model fallback reproduce across at least three seeds.
- [ ] Teacher plus Stage 1 post-processor meets Macro-F1 >= 0.60, kappa >= 0.50 and accuracy >= 0.60 on the locked checkpoint.
- [ ] Pre-post-processing teacher results are reported separately.
- [ ] REM, Deep, worst-domain, calibration and transition failures are reviewed.
- [ ] Major front-end, augmentation, loss, temporal and ensemble contributions are ablated.
- [ ] Teacher, calibration, preprocessing and output schema are frozen.
- [ ] Teacher logits/embeddings/confidence/quality can be generated deterministically.
- [ ] Golden outputs and artifact hashes pass verification.
- [ ] Stage 1 report and release manifest are approved by all researchers.

### 16.3 Gate C - Student release candidate

- [ ] Student family is selected using accuracy plus measured device results.
- [ ] Hard-label, logit, feature, robustness and applicable sequence KD steps are ablated.
- [ ] Structured pruning and QAT contributions are measured independently.
- [ ] Training, export and mobile front ends pass golden parity tests.
- [ ] Full compiled package is < 50 MB.
- [ ] Peak RAM is <= 250 MB on all target devices.
- [ ] Approved latency, battery and thermal thresholds pass.
- [ ] Eight-hour operation and background interruption tests pass.
- [ ] No raw audio is transmitted or exposed through telemetry.
- [ ] Embedding transfer, if any, has privacy approval and a logits-only comparison.

### 16.4 Gate D - Final acceptance

- [ ] Final hybrid Macro-F1 is >= 0.60.
- [ ] Final hybrid Cohen's kappa is >= 0.50.
- [ ] Final hybrid accuracy is >= 0.60.
- [ ] Edge-only and hybrid per-class results are published.
- [ ] Subject-level uncertainty and domain/device/noise slices are published.
- [ ] Teacher-student gap is documented.
- [ ] Model, package/API specification and end-to-end pipeline are delivered.
- [ ] Performance and technical reports cover methods, ablations, limitations and reproduction.
- [ ] Artifact hashes, versions, approvals and rollback target are recorded.
- [ ] Data rights, privacy and intended-use limitations are approved.

## 17. Failure Modes and Required Responses

| Failure mode | Symptom | Required response |
|---|---|---|
| Metric/split drift | Results from researchers are no longer comparable | Stop leaderboard updates, version the fix, rerun baseline/top candidates |
| Subject or channel leakage | Unusually high validation performance | Mark affected runs invalid, repair manifests, rerun from clean splits |
| Test overuse | Frequent test checks guide tuning | Stop access, record incident, make decisions on validation only |
| Noise chasing | Small single-seed gains drive integration | Require confirmatory seeds and meaningful-difference rule |
| Lane duplication | Two owners run nearly identical work | Compare proposals, merge effort or separate one controlled factor |
| Negative-result loss | Failed ideas are repeated | Require immediate failed/stopped logging and weekly audit |
| Monolithic code | Winning method cannot be combined or reproduced | Refactor into tested components before integration |
| Teacher drift | Cached KD targets come from mixed teachers | Invalidate mixed cache; regenerate from one frozen version |
| Distillation dominated by Light | Overall accuracy rises while REM/Deep collapse | Rebalance losses/cache, track per-class KD and gate on class behavior |
| Synthetic robustness illusion | Augmented validation improves, real-home data worsens | Reject policy; prioritize real-domain slices and controlled curricula |
| QAT/export mismatch | Simulated INT8 passes but phone output fails | Compare exported/mobile logits; layer sensitivity and selective precision |
| FLOPs/device mismatch | Small-FLOP model is slow or power-hungry | Benchmark operators on device and change architecture/runtime |
| Package accounting error | Model weights fit but compiled artifact exceeds 50 MB | Count runtime/assets from the first Stage 2 spike and budget explicitly |
| Eight-hour termination | OS kills the process or memory grows | Profile lifecycle/memory, reuse buffers, duty cycle, test interruptions |
| Embedding privacy risk | Derived features expose identity or content | Prefer logits, minimize/test embeddings, restrict and encrypt access |
| License uncertainty | Research dataset cannot support product training | Separate research and production pipelines; obtain approval or replace data |

## 18. Setup Checklist

Complete before substantial Stage 1 compute:

- [ ] `TEAM.md` lists real names, `r01`-`r05`, primary lanes, shared roles and backups.
- [ ] Repository structure and branch protection exist.
- [ ] Raw data is outside ordinary Git and access is controlled.
- [ ] Dataset cards, rights status and data manifests exist.
- [ ] Canonical labels, group splits and overlap audit exist.
- [ ] Evaluation harness, version and result schema exist.
- [ ] Baseline results exist; all five researchers pass the fixture and at least two reproduce the full baseline.
- [ ] Confirmatory seeds and meaningful-difference decision are recorded.
- [ ] Experiment/decision/test-access/release registries exist.
- [ ] Five researcher work plans and status files exist.
- [ ] Stage 1 experiment lanes and initial compute budget are assigned.
- [ ] Artifact storage, hashing, retention and access procedures are tested.
- [ ] The team has read Part I and can explain the Stage 1-to-Stage 2 handoff.

Complete before Stage 2 optimization:

- [ ] Gate B passes and the teacher release manifest is signed.
- [ ] Teacher cache generation and checksum validation are tested.
- [ ] Target operating systems, devices and responsible owners are named.
- [ ] Runtime/export feasibility spike is complete.
- [ ] Full package and peak-RAM accounting methods are defined.
- [ ] Golden front-end and inference parity fixtures exist.
- [ ] Latency, battery and thermal thresholds have an owner and decision date.
- [ ] Embedding transfer defaults to off until privacy/performance evidence is reviewed.

## 19. Definition of Done

The team is done only when the delivered system is scientifically valid, operationally reproducible and deployable:

- Stage 1 produced the strongest validated and calibrated teacher within the agreed search budget.
- Stage 2 transferred that behavior into a mobile-efficient model and quantified every major compression trade-off.
- The full compiled package and measured runtime meet the phone constraints.
- The edge-only and hybrid paths are independently testable and versioned.
- The final locked-set metrics meet the research-plan thresholds.
- An authorized researcher can reproduce training, export, evaluation and deployment from the released documentation and manifests.
- Data rights, privacy boundaries, known limitations and rollback procedures are explicit.

Anything less is an experiment or prototype, not the completed project.
