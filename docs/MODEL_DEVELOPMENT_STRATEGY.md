# Sleep Stage Model Development Strategy

**Project:** Real-World Sleep Stage Prediction for On-Device Smartphone Application  
**Target classes:** Wake, REM, Light (N1 + N2), Deep (N3)  
**Program structure:** Stage 1 - maximize teacher performance; Stage 2 - distill and deploy a lightweight student  
**Target completion:** 27 November 2026  
**Strategy version:** 1.0 - startup-ready baseline  
**Final alignment review:** 8 September 2026  
**Team execution procedures:** [PROJECT_HANDBOOK.md](./PROJECT_HANDBOOK.md)  
**Basis:** `연구개발계획서_서울대학교_v2.pdf`

## 1. Purpose

This document converts the research plan into an executable model-development strategy with two deliberately different optimization stages:

1. **Stage 1: High-performance teacher.** Maximize predictive performance and robustness without imposing smartphone model-size or latency limits. The teacher may be a large model, a temporal model, or an ensemble, provided that it produces stable class probabilities and embeddings that can be transferred to a smaller model.
2. **Stage 2: On-device student.** Transfer the teacher's behavior into a compact model, then optimize and verify that model on representative smartphones under the project's hard resource limits.

The separation is important. Stage 1 establishes the best available prediction target; Stage 2 preserves as much of that performance as possible while satisfying deployment constraints. Mobile constraints must not prematurely restrict teacher exploration, and teacher-specific complexity must not leak into the final on-device package.

## 2. Project Outcomes and Non-Negotiable Constraints

### 2.1 Functional scope

- Predict one sleep stage every 30 seconds from smartphone-recorded nocturnal audio.
- Use four classes: **Wake, REM, Light, Deep**.
- Treat N1 and N2 as Light and N3 as Deep when mapping source labels.
- Focus on sleep staging. Obstructive sleep apnea diagnosis is a possible future extension, not a primary output of this project.
- Keep raw audio on the device. If the hybrid path is enabled after waking, transmit only the minimum approved derived data, such as probability sequences, embeddings, timestamps, and quality indicators.

### 2.2 Contract-level performance criteria

The final hybrid system must meet all of the following on the preselected, subject-disjoint, held-out real-world smartphone evaluation set:

| Metric | Minimum requirement |
|---|---:|
| Macro-averaged F1 | >= 0.60 |
| Cohen's kappa | >= 0.50 |
| Accuracy | >= 0.60 |

Macro-F1 is the primary model-selection metric because it gives equal weight to Wake, REM, Light, and Deep despite class imbalance. Kappa and accuracy are mandatory secondary gates, not substitutes for Macro-F1.

### 2.3 On-device constraints

| Constraint | Acceptance limit |
|---|---:|
| Compiled inference package / binary | < 50 MB |
| Peak RAM during inference | <= 250 MB |
| Operating pattern | Repeated 30-second epoch inference over an 8-hour night |
| Raw-audio transmission | Prohibited by default |

The 50 MB limit applies to the deployable artifact, not only the neural-network weight file. Runtime, operators, lookup tables, and preprocessing assets must therefore be counted. Battery drain, thermal behavior, and latency do not yet have contract-level numeric limits; they must be baselined on target devices early in Stage 2 and converted into device-specific acceptance thresholds before release-candidate testing.

## 3. End-to-End System Boundary

The system has two prediction paths, both of which must be reported:

1. **Edge-only path:** native audio capture and feature extraction -> student model -> per-epoch stage probabilities and embeddings.
2. **Hybrid final path:** the edge outputs collected across the night -> optional server-side temporal post-processing -> final stage sequence.

The edge-only path is necessary for deployment diagnosis and privacy-preserving operation. The hybrid path is the contract-level performance path because a bidirectional sequence model can correct implausible transitions using the full-night context after the user wakes.

Stage 1 may use full-night context and ensembles to establish an upper bound. Stage 2 must explicitly distinguish:

- information available to the on-device model at inference time;
- information used only by the post-wake cloud sequence model; and
- information used only during training.

This prevents accidental evaluation leakage and makes the deployment claim reproducible.

## 4. Shared Data and Evaluation Foundation

The same frozen data definitions and evaluation implementation must be used by both stages. Without this, teacher-to-student comparisons are not meaningful.

### 4.1 Data sources

| Source | Intended use | Restrictions / notes |
|---|---|---|
| PSG-Audio | Supervised training, validation, and research evaluation | 212 sleep-lab participants; synchronized PSG and audio; microphone, tracheal, and snore channels. Split by participant/night, never by epoch. |
| Multimodal OSA | Supervised training, cross-device/domain validation | 50 sleep-apnea patients; 400+ hours; mobile-phone and recorder audio. Split by participant/night and preserve device metadata. |
| Internal PSG data | Research-only sensitivity and expected-improvement analysis | Must not be used in a commercial release unless governance and rights are separately cleared. |
| Unlabeled sleep audio | Self-supervised learning, consistency training, and distillation | Use only data with appropriate consent, privacy controls, and permitted use. Availability may follow service launch; the pipeline must also work without it. |
| General licensed audio/noise/RIR/MIR assets | Robustness pretraining and augmentation | Track provenance, license, sampling policy, and version. |

Before data is used, complete a dataset license and commercial-use review. Public research availability must not be assumed to authorize commercial deployment.

### 4.2 Canonical label and epoch policy

- Convert source annotations to 30-second epochs on a single canonical timeline.
- Map Wake -> Wake, REM -> REM, N1/N2 -> Light, and N3 -> Deep.
- Define deterministic rules for movement, unknown, transition, and partially labeled epochs. Exclude or mask them rather than silently coercing them into a valid class.
- Record label-source version and any scoring-manual differences.
- When multiple audio channels exist for the same night, keep every channel from that night in the same split.
- Fit normalization statistics on the training split only.

### 4.3 Frozen split hierarchy

Use a group-aware split keyed by participant and recording night:

- **Training:** model fitting, augmentation, self-supervision, and pseudo-labeling.
- **Validation:** architecture selection, hyperparameter selection, calibration, and ablation.
- **Locked test:** used only at the formal Stage 1 freeze and final Stage 2 release gate.
- **Target-domain held-out set:** real smartphone recordings in representative home-noise conditions; fixed before final optimization.

If sample size permits, include an external-dataset or leave-one-dataset-out evaluation to measure domain generalization. No epoch, recording, participant, augmented derivative, or teacher-generated label may cross split boundaries.

### 4.4 Evaluation outputs

Every formal run must produce:

- Macro-F1, Cohen's kappa, and accuracy;
- per-class precision, recall, F1, and support;
- confusion matrix;
- results by dataset, device/channel, sex when available, OSA severity when available, noise type, and SNR bucket;
- mean and standard deviation across at least three seeds for model-selection claims;
- subject-level bootstrap confidence intervals for final checkpoints;
- expected calibration error and reliability plots for teacher probabilities used in distillation;
- sleep-transition violation rate and stage-duration plausibility for sequence outputs;
- inference latency, model/package size, peak RAM, and, in Stage 2, energy/thermal measurements.

The test set remains locked between stage gates. Exploratory decisions are made on validation results only.

## 5. Stage 1 - Maximize Teacher Model Performance

### 5.1 Objective

Build the most accurate, robust, and well-calibrated audio-based sleep-stage predictor possible with the available data. Stage 1 is **not constrained by smartphone model size, RAM, or latency**. Compute-heavy sequence context, foundation models, multi-view inputs, and ensembles are permitted.

The teacher must still expose a stable distillation interface:

- calibrated logits or probabilities for each 30-second epoch;
- one or more intermediate feature embeddings;
- confidence and input-quality indicators;
- optional sequence-level posterior after temporal processing; and
- a documented mapping between inputs, epochs, outputs, and class order.

### 5.2 Stage 1 work packages

#### WP1.1 - Reproducible baselines

Establish simple baselines before expensive exploration:

- log-Mel + 2D CNN epoch classifier;
- pretrained audio encoder with a linear classification head;
- epoch encoder plus Bi-LSTM temporal head;
- majority-class and transition-prior baselines for sanity checks.

The baseline phase validates label mapping, split integrity, metric implementation, and feature alignment. It also quantifies the gain attributable to each later technique.

#### WP1.2 - Front-end and acoustic representation search

Run controlled experiments over:

- sample rate and band limit appropriate to smartphone microphones;
- log-Mel versus PCEN-Mel representations;
- single-resolution versus multi-resolution spectrograms;
- FFT window, hop, Mel-bin count, normalization, and dynamic-range clipping;
- 30-second epoch input versus short neighboring-epoch context;
- waveform and spectrogram encoders when compute allows.

Teacher-only multi-view models may combine representations. However, at least one teacher branch must accept the same canonical feature family planned for the student; this makes feature-level distillation practical and reduces train-deploy mismatch.

#### WP1.3 - Foundation-model adaptation

Prioritize the foundation backbones named in the research plan:

- **OPERA** as a respiratory/acoustic representation candidate;
- **AudioMAE** as a general masked-audio representation candidate.

Compare frozen-probe, partial fine-tuning, parameter-efficient adaptation, and full fine-tuning where compute and data volume permit. Add one strong non-foundation sequence baseline so improvements are not attributed to pretraining without evidence.

When unlabeled sleep audio is available, perform domain self-supervised adaptation before supervised fine-tuning. Candidate objectives include masked spectrogram reconstruction, contrastive consistency between clean and augmented views, and teacher-student consistency. Do not use locked validation or test audio in self-supervised adaptation unless the evaluation protocol explicitly allows transductive learning; the default is that it does not.

#### WP1.4 - Real-world noise and device robustness

Build an augmentation engine that composes, rather than merely samples, realistic conditions:

- SpecAugment time/frequency masking;
- room impulse response simulation;
- random microphone impulse response augmentation;
- dynamic SNR noise injection using HVAC, traffic, white/pink noise, bedding movement, fan, and speech-like distractors;
- gain variation, clipping, automatic-gain-control simulation, codec artifacts, band limiting, packet/dropout-like gaps, and microphone occlusion;
- channel selection and resampling artifacts matching source datasets and target devices.

Use a curriculum: begin with plausible moderate corruption, then widen the SNR and device distribution after the model has learned sleep-related structure. Validate each augmentation family separately and in combination. Reject augmentations that improve synthetic-noise performance while degrading clean or real-home performance.

#### WP1.5 - Supervised objectives and imbalance control

Compare the following under identical splits:

- cross-entropy with class-balanced sampling or weighting;
- focal loss for hard and underrepresented stages;
- label smoothing to reduce overconfidence under label uncertainty;
- soft targets near annotated stage boundaries;
- supervised contrastive loss on embeddings;
- auxiliary quality/SNR/domain prediction heads.

Use domain-adversarial training with a gradient reversal layer when domain labels are reliable. Candidate domain labels include dataset, recording device/channel, microphone simulation, room simulation, and SNR bucket. The goal is a stage-discriminative but domain-invariant representation. Track stage performance and domain-probe accuracy together; suppressing domain information is useful only if sleep-stage performance and real-domain transfer improve.

#### WP1.6 - Temporal modeling and physiological consistency

Evaluate context at two levels:

1. **Joint teacher:** an epoch encoder followed by a Bi-LSTM, temporal convolutional network, Conformer, or lightweight Transformer over a sequence of epochs.
2. **Post-processing teacher:** a separate server-side Bi-LSTM baseline, then compare a temporal convolutional model, transformer encoder, CRF/HMM constraint layer, or a learned transition smoother.

The temporal module should improve stage continuity without erasing genuine short awakenings or REM transitions. Report both frame-level metrics and transition/duration plausibility. Any model that uses future epochs must be labeled post-wake/non-causal and must not be represented as a real-time edge model.

#### WP1.7 - Semi-supervised learning and pseudo-labeling

If suitable unlabeled audio is available:

- generate calibrated teacher pseudo-labels;
- filter or weight them by confidence and input quality;
- balance accepted pseudo-labels by predicted class and domain;
- use weak/strong augmentation consistency;
- compare hard pseudo-labels with soft distributions;
- audit confirmation bias, particularly for REM and Deep.

Unlabeled data is never added to validation or test. A pseudo-label experiment is accepted only if gains persist on real, non-augmented held-out recordings.

#### WP1.8 - Ensemble, calibration, and convergence

Near the end of Stage 1:

- retain complementary models, not merely the top replicas;
- test probability averaging, weighted averaging, and stacking using out-of-fold predictions;
- use model soups when checkpoints share an architecture and optimization path;
- calibrate the selected teacher or ensemble with validation-only temperature scaling or another justified method;
- measure ensemble diversity through disagreement and per-class error overlap.

The final teacher may be an ensemble because it will run offline during distillation. A single-model teacher must also be retained as a lower-cost fallback.

### 5.3 Stage 1 experiment priority

| Priority | Experiment family | Decision to make |
|---|---|---|
| P0 | Split, label, metric, and simple baseline validation | Can all subsequent numbers be trusted? |
| P0 | OPERA and AudioMAE adaptation baselines | Which representation transfers best to sleep audio? |
| P1 | Noise/RIR/MIR/device augmentation | Which corruption policy improves real-home generalization? |
| P1 | Sequence context and post-processing | How much performance comes from temporal information? |
| P1 | Imbalance, boundary, and domain-adversarial objectives | Which losses improve REM/Deep without harming calibration? |
| P2 | Self-supervised domain adaptation and pseudo-labeling | Does unlabeled data provide genuine held-out gains? |
| P2 | Ensemble and calibration | What is the strongest stable distillation teacher? |

### 5.4 Stage 1 exit gate

Stage 1 ends only when all of the following are true:

- the evaluation harness, split manifest, label mapping, and preprocessing version are frozen;
- the best teacher and a lower-cost fallback are reproduced across at least three seeds;
- performance has plateaued under the agreed search budget, with major negative results documented;
- the calibrated teacher plus its Stage 1 temporal post-processor is evaluated once on the locked checkpoint set and meets Macro-F1 >= 0.60, kappa >= 0.50, and accuracy >= 0.60; the pre-post-processing teacher result is reported separately;
- probability calibration is complete and class-specific failure modes are documented;
- teacher logits, embeddings, confidence, and quality outputs can be generated deterministically for labeled and permitted unlabeled data;
- a teacher model card, training configuration, checkpoint, and inference interface are frozen for Stage 2.

The recommended internal target is to maintain a performance margin above the contract minima so that compression has room to lose a small amount of accuracy. This is a planning margin, not a replacement for the formal acceptance criteria.

## 6. Stage 2 - Build and Deploy the Knowledge-Distilled Student

### 6.1 Objective

Produce a smartphone-deployable student that preserves the teacher's decision structure while satisfying the hard size and memory constraints and operating reliably for an eight-hour night. Stage 2 optimization is multi-objective:

1. maximize Macro-F1 and kappa;
2. minimize the gap to the frozen teacher;
3. meet package size and peak RAM limits;
4. minimize latency, energy use, and thermal impact on real target devices; and
5. preserve stable behavior across microphone, room, and noise domains.

### 6.2 Student architecture search

Start with mobile-friendly operators that map efficiently to phone accelerators:

- customized MobileNetV3-Small and MobileNetV3-Large variants;
- EfficientNet-Lite variants;
- a compact depthwise-separable CNN as a latency-oriented fallback;
- an optional small causal temporal convolution or gated recurrent head when its measured gain justifies its memory and energy cost.

Search width multiplier, depth, expansion ratio, input resolution, Mel-bin count, and embedding dimension. Use actual device latency and memory, not parameter count or FLOPs alone, to choose the winner. Avoid operators with poor support in the selected mobile runtime even when desktop benchmarks look favorable.

### 6.3 Distillation design

Freeze the calibrated teacher before student tuning. Cache teacher outputs with dataset, preprocessing, teacher-checkpoint, and calibration versions.

A default supervised distillation objective is:

```text
L = alpha * L_hard
  + beta  * T^2 * KL(softmax(z_teacher / T), softmax(z_student / T))
  + gamma * L_feature
  + delta * L_sequence
  + eta   * L_robustness
```

Where:

- `L_hard` is class-balanced cross-entropy or focal loss against PSG labels;
- the KL term transfers softened class relationships at temperature `T`;
- `L_feature` aligns selected teacher and projected student embeddings;
- `L_sequence` transfers temporal/posterior behavior when comparable context is available; and
- `L_robustness` enforces prediction consistency across clean/noisy or cross-device views.

Tune the coefficients and temperature on validation only. For unlabeled examples, omit `L_hard` and use confidence/quality-weighted soft-target, feature, and consistency terms. Apply per-class controls so abundant Light epochs do not dominate distillation. Low-confidence teacher outputs should be down-weighted or excluded, especially near stage boundaries or under severe acoustic corruption.

Test distillation in increasing complexity:

1. hard labels only;
2. logit distillation;
3. logit + feature distillation;
4. logit + feature + robustness consistency;
5. optional sequence/transition distillation;
6. ensemble-teacher distillation.

Each step must earn its added training and deployment complexity through a reproducible gain.

### 6.4 Compression and quantization sequence

Apply optimizations in an order that preserves attribution:

1. choose a hardware-efficient dense student architecture;
2. complete knowledge distillation in floating point;
3. apply structured channel pruning and fine-tune, if pruning produces real device gains;
4. simplify or fuse unsupported operators;
5. perform integer quantization-aware training with representative clean/noisy/device data;
6. use mixed precision only for layers whose integer conversion causes a verified accuracy collapse;
7. compile using the release runtime and repeat accuracy checks on exported outputs;
8. benchmark on target devices and adjust width/input resolution if constraints are missed.

Unstructured sparsity is not a default optimization because it often reduces parameter count without improving smartphone latency. It should be used only when the target runtime and hardware demonstrate a measurable benefit.

### 6.5 Front-end parity and mobile runtime

Training and deployment feature extraction must be numerically matched:

- define sample rate, framing, window, FFT, Mel filters, log/PCEN transform, normalization, clipping, padding, and timestamp behavior in a versioned feature specification;
- build golden audio fixtures and expected feature tensors;
- require desktop and mobile front ends to agree within a declared numerical tolerance;
- record how interruptions, missing samples, clock drift, device sample-rate conversion, and microphone permission loss are handled.

Use native low-power audio capture and DSP paths, consistent with iOS vDSP and Android Oboe/AAudio or their selected production equivalents. Buffer a 30-second epoch, wake the accelerator for a short inference burst, save derived outputs, then return to low-power operation.

The mobile inference interface should return:

- four logits/probabilities in a fixed class order;
- predicted stage;
- timestamp and epoch index;
- compact embedding when the hybrid mode is enabled;
- input-quality/confidence indicator; and
- model, feature-specification, and calibration versions.

### 6.6 Hybrid temporal post-processing

Retain the server-side temporal model as a separately versioned component. Its input is the student's nightly sequence of logits and, only when approved and necessary, compact embeddings and quality flags. Begin with the planned Bi-LSTM and compare alternatives only if they improve held-out performance or reduce operational cost.

The post-processor must:

- handle missing or low-quality epochs;
- avoid impossible or highly implausible transitions without over-smoothing;
- output an aligned four-class sequence;
- preserve the original edge probabilities for audit and ablation;
- be evaluated with and without embeddings so privacy/performance trade-offs are explicit.

Because embeddings can retain sensitive information, they must not be treated as automatically anonymous. Perform a privacy review, minimize dimension and retention, encrypt transmission/storage, test whether logits alone are sufficient, and document deletion policies.

### 6.7 Device verification matrix

Before the release candidate, test at least one lower-tier and one target-tier device for each supported operating system. The final matrix should include:

| Test | Measurement |
|---|---|
| Functional parity | Mobile logits versus exported desktop-runtime logits on golden fixtures |
| Package size | Full compiled artifact including model and required runtime components |
| Peak RAM | Cold start and steady-state inference under background execution |
| Latency | Median and p95 per 30-second epoch, including feature extraction |
| Eight-hour endurance | Completion rate, battery change, thermal state, crashes, forced termination |
| Interruptions | Calls, alarms, app suspension/resume, audio-route changes, low storage, low battery |
| Acoustic robustness | Distance, orientation, bedding occlusion, fan/HVAC, traffic, speech/TV, different rooms |
| Privacy | No raw-audio network transmission; logs and crash reports contain no audio payload |

Convert latency, battery, and thermal observations into formal release thresholds after the initial hardware spike. Freeze those thresholds before the final optimization round.

### 6.8 Stage 2 exit gate

Stage 2 is complete only when:

- the compiled on-device package is < 50 MB;
- measured peak RAM is <= 250 MB on every target test device;
- the student completes the defined eight-hour duty-cycle test without unacceptable crash, forced termination, battery, or thermal behavior;
- exported/mobile inference matches the reference implementation within tolerance;
- the final hybrid output meets Macro-F1 >= 0.60, kappa >= 0.50, and accuracy >= 0.60 on the locked real-world smartphone set;
- edge-only and hybrid metrics, including per-class F1, are both reported;
- the teacher-student performance gap and the contribution of distillation, pruning, quantization, and post-processing are quantified by ablation;
- the model, runtime/API specification, end-to-end training/inference code, evaluation report, and technical documentation are delivered.

A recommended engineering goal is to keep student Macro-F1 within three percentage points of the frozen teacher and prevent any single class from suffering a disproportionate collapse. This is an internal guardrail and may be revised using observed teacher performance and device limits.

## 7. Two-Stage Governance and Decision Rules

### 7.1 What is frozen at the handoff

The Stage 1 to Stage 2 handoff freezes:

- split manifest and evaluation version;
- label mapping and feature/epoch definitions;
- selected teacher checkpoint or ensemble;
- teacher calibration;
- teacher output schema and cached-output version;
- target mobile operating systems, representative devices, and initial runtime choice;
- Stage 2 resource accounting method.

Teacher improvements after the freeze enter Stage 2 only through a versioned change decision followed by regeneration of all distillation targets. Silent teacher replacement is prohibited.

### 7.2 Experiment acceptance rules

- A single-seed gain is provisional; decisions require at least three seeds unless the experiment is only a feasibility check.
- Predefine a practical significance threshold; differences below it are treated as noise.
- Change one major factor at a time before combining techniques.
- Log failed and negative experiments so they are not repeated.
- Select models on validation, not test.
- Report resource and accuracy regressions together in Stage 2.
- A lower-FLOP model is not considered faster until confirmed on device.
- A quantized model is not accepted until the exported/mobile artifact, rather than the training simulation alone, passes accuracy checks.

### 7.3 Stage comparison dashboard

Maintain a single comparison table with at least these columns:

| Category | Fields |
|---|---|
| Identity | experiment ID, code revision, data/split version, seed, teacher version |
| Accuracy | Macro-F1, kappa, accuracy, per-class F1, calibration |
| Robustness | dataset/device/noise/SNR breakdown and worst-group result |
| Temporal quality | transition violation rate, duration plausibility |
| Deployment | weight size, compiled package size, peak RAM, median/p95 latency, energy |
| Reproducibility | config hash, checkpoint, runtime/export version, result artifact |

## 8. Proposed Schedule

This schedule preserves the research plan's broad order - data and baseline work first, performance optimization next, and lightweight deployment last - while making the two formal stages explicit.

| Period | Stage | Main activities | Gate / output |
|---|---|---|---|
| June-July 2026 | Stage 1 | Data acquisition, provenance, label harmonization, subject-level splits, frozen evaluation harness | Data and evaluation specification |
| July-August 2026 | Stage 1 | Baselines, front-end search, OPERA/AudioMAE adaptation, initial temporal model | Reproduced baseline and candidate leaderboard |
| August-September 2026 | Stage 1 | Robustness, domain adaptation, loss/imbalance work, SSL/semi-supervision where available, temporal post-processing, ensemble and calibration | Frozen high-performance teacher and Stage 1 report |
| Early October 2026 | Stage 2 | Runtime spike, student architecture benchmark, front-end parity fixtures | Selected student/runtime/device profile |
| October-early November 2026 | Stage 2 | Logit/feature/sequence distillation, structured pruning, QAT, iterative on-device profiling | Student release candidate |
| November-27 November 2026 | Stage 2 | Hybrid integration, locked evaluation, eight-hour device tests, packaging and documentation | Final model, package/API spec, code, evaluation report, technical document |

If the project begins later than the calendar shown, retain the ordering and gates rather than compressing away the evaluation freeze or device-validation period.

## 9. Risks and Mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Small labeled dataset | High variance and overfitting | Subject-level splits, transfer learning, multi-seed evaluation, strong but validated augmentation, SSL and distillation |
| Domain gap between sleep lab and bedroom | Good lab scores but poor product performance | Smartphone/device augmentation, RIR/MIR, real-home held-out set, leave-one-domain-out analysis |
| Severe class imbalance | Light dominates; REM/Deep failures hidden by accuracy | Macro-F1 primary, class-balanced losses/sampling, per-class gates and error analysis |
| Annotation boundary noise | Model punished for ambiguous transitions | Soft boundary targets, context models, calibrated probabilities, tolerance analysis reported separately from official metrics |
| Teacher overconfidence | Poor soft targets and confirmation bias | Validation-only calibration, confidence weighting, pseudo-label audits |
| Teacher-student input mismatch | Distillation fails despite a strong teacher | Maintain a compatible teacher branch, align epochs/features, version cached outputs |
| Quantization accuracy loss | Student misses performance threshold | QAT, representative calibration data, selective mixed precision, layer-wise sensitivity testing |
| Model meets FLOPs target but is slow | Missed battery/latency goal | Early benchmarking on actual phones; choose supported operators |
| Runtime/package overhead breaks 50 MB limit | Nominally small network cannot ship | Count full compiled package from first Stage 2 spike; allocate explicit size budget |
| Eight-hour OS termination | Incomplete nights and poor user experience | Epoch duty cycling, memory reuse, background-mode tests, interruption recovery |
| Embedding privacy leakage | Privacy claim invalidated | Prefer logits, minimize embeddings, privacy testing, encryption, retention controls, explicit consent/governance |
| Dataset licensing blocks commercialization | Model cannot legally ship | License review before use; separate research benchmarks from production-training data |
| Post-processing hides weak edge performance | Product depends excessively on cloud | Always report edge-only and hybrid results; maintain a logits-only fallback |

## 10. Required Deliverables

### Stage 1 deliverables

- frozen dataset/split/label and preprocessing specification;
- reproducible baselines and experiment ledger;
- best single teacher plus calibrated ensemble, if used;
- teacher checkpoint(s), inference code, output schema, and model card;
- robustness, calibration, sequence, subgroup, and ablation report;
- cached teacher targets or a deterministic job for regenerating them;
- Stage 1 gate report and approved student requirements.

### Stage 2 and final deliverables

- PyTorch-based lightweight sleep-stage model and export artifact;
- mobile inference API/runtime specification suitable for application integration;
- end-to-end code for data loading, preprocessing/feature extraction, training, distillation, export, inference, post-processing, and evaluation;
- golden front-end and inference parity fixtures;
- device benchmark report covering package size, RAM, latency, endurance, battery, and thermal behavior;
- performance evaluation report with datasets, noise methods, metrics, subgroup results, ablations, limitations, and expected value of future proprietary PSG data;
- technical document covering architecture, training, reproduction, deployment, privacy boundary, and rollback/versioning procedures.

## 11. Final Acceptance Checklist

### Scientific validity

- [ ] Four-stage label mapping is deterministic and documented.
- [ ] Splits are participant/night disjoint and immutable.
- [ ] All formal model comparisons use the same evaluation version.
- [ ] Final claims include multi-seed variation and subject-level uncertainty.
- [ ] Test and target-domain held-out sets were not used for tuning.
- [ ] Edge-only and hybrid results are separately reported.

### Teacher readiness

- [ ] Teacher performance has plateaued under the agreed search budget.
- [ ] The frozen teacher is calibrated and reproducible.
- [ ] Logits, embeddings, confidence, and quality outputs are versioned.
- [ ] Teacher errors by class, dataset, noise, and device are understood.

### Student and deployment readiness

- [ ] Distillation contribution is isolated by ablation.
- [ ] Exported/mobile logits pass parity tests.
- [ ] Compiled package is < 50 MB.
- [ ] Peak RAM is <= 250 MB on every target device.
- [ ] Eight-hour duty-cycle and interruption tests pass.
- [ ] Final hybrid Macro-F1 is >= 0.60.
- [ ] Final hybrid Cohen's kappa is >= 0.50.
- [ ] Final hybrid accuracy is >= 0.60.
- [ ] No raw audio is transmitted or exposed through logs/crash reports.
- [ ] Data rights and privacy controls are approved for the intended use.

## 12. Summary of the Strategy

Stage 1 should pursue the strongest credible teacher using foundation-model adaptation, realistic acoustic augmentation, domain-invariant training, temporal modeling, semi-supervision when permitted, ensembling, and calibration. Its product is not a phone model; it is a stable, well-understood source of high-quality predictions and representations.

Stage 2 should treat the frozen teacher as the behavioral target, select a hardware-efficient CNN through real-device benchmarking, transfer logits and features, then apply structured compression and quantization-aware training. Final success requires both predictive performance and verified smartphone operation. The hybrid post-processor may maximize final accuracy, but edge-only quality, privacy, package size, memory, and eight-hour reliability remain first-class acceptance criteria.
