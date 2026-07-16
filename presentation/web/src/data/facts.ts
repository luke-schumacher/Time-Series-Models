/**
 * SINGLE SOURCE OF TRUTH for every number and claim on a slide.
 *
 * Provenance:
 *  - `deck`  — presentation/main.tex + presentation/svgs/ (the June-2026 handover deck)
 *  - `thesis`— agentic-infra-copilot/results/ablation/thesis_summary.json (final run,
 *              2026-04-09; the repo README table is a STALE earlier run)
 *  - `repo`  — AlternatingPipeline validation history (VERIFY before presenting; see TODOS)
 *
 * Do not put numbers on slides that are not in this file.
 */

export const PRESENTER = 'Luke Schumacher · SHS DI D&A';
export const PRESENTED = 'July 2026';
export const TEAM = 'Team of 4 — Luke Schumacher · Navneet · Martina · Georg (team lead)';
export const SEBASTIAN_CREDIT = 'With thanks to Sebastian — data access & feedback';

/** Rendered inside TodoChip, which supplies the “TODO” label itself. */
export const TODOS = {
  screenshots: 'Drop-in slot — see README',
} as const;

/* ---------------------------------------------------------------- problem */

export const PROBLEM = {
  scanRange: '30 s – 45 min',
  baselineLossUsd: 25_000,
  baselineLossSource: 'Industry analysis',
  waitReductionPct: 44.8,
  utilizationGainPct: 14.5,
  literatureSource: 'Silva-Aravena et al. (2025)',
  failureWarning: '2 weeks',
  durationRanges: [
    { region: 'HEAD (Scout)', range: '0.5 – 2 min', min: 0.5, max: 2 },
    { region: 'HEAD (Full)', range: '6 – 12 min', min: 6, max: 12 },
    { region: 'SPINE (Cervical)', range: '8 – 18 min', min: 8, max: 18 },
    { region: 'SPINE (Full)', range: '15 – 35 min', min: 15, max: 35 },
    { region: 'PELVIS', range: '10 – 25 min', min: 10, max: 25 },
    { region: 'ABDOMEN', range: '12 – 30 min', min: 12, max: 30 },
  ],
  rootCause:
    'Static scheduling treats stochastic processes as deterministic — the wrong model for MRI workflows.',
} as const;

export const FOUR_TWINS = [
  { name: 'Device Twin', desc: 'Hardware monitoring & maintenance', active: false },
  { name: 'Patient Twin', desc: 'Care pathway & clinical outcomes', active: false },
  { name: 'Factory Twin', desc: 'Production & supply chain', active: false },
  { name: 'Customer Twin', desc: 'Workflow simulation & schedule optimization', active: true },
] as const;

export const GAP_TABLE = [
  ['Manual, fixed-slot scheduling', 'Dynamic scheduling from real-time simulation', 'Generative synthetic day with μ ± σ per slot'],
  ['Reactive maintenance after failure', '2-week advance failure warning', 'Tube-life + arcing signal monitoring model'],
  ['Post-hoc utilisation reports', 'Forward-looking capacity simulation', 'Customer Twin: simulate any future day'],
  ['Coil ROI guessed at point-of-sale', 'Pre-purchase ROI simulation per site', 'Counterfactual day simulation with/without upgrade'],
  ['Paper-based technician training', 'Immersive VR practice environment', 'Unity / Meta Quest simulation from digital twin'],
  ['Manual R&D feedback triage', 'Automated semantic pain extraction', 'MRRT LLM Insight Agent over customer comments'],
] as const;

/* ----------------------------------------------------------------- system */

export const DATA_FOUNDATION = {
  scanners: 40,
  window: 'Apr – May 2024',
  windowLen: '2 months',
  vocab: 18,
  vocabDesc: 'MRI_CCS · EXU · MSR · MPT · FRR',
  regions: 11,
  regionsDesc: 'HEAD · NECK · CHEST · ABD · PELVIS · SPINE …',
  seqTypes: 15,
  seqTypesDesc: 'TSE · SPACE · HASTE · GRE · EPI · TFL …',
  serials: '141049 · 155687 · 175693 · 175727 · 175832 · …',
  eventsNote:
    'Each scanner CSV holds thousands of timestamped hardware events: coil changes, patient tracking, scan markers, duration telemetry.',
} as const;

export const WORKSTREAMS = [
  {
    n: 1,
    name: 'Alternating Pipeline',
    points: ['Generative transformer models', 'Exchange ↔ Examination alternation', 'Orchestrator for day sequencing', 'μ ± σ duration per token', 'Output: full synthetic day CSV'],
    mine: true,
  },
  {
    n: 2,
    name: 'MRRT Insight Agent',
    points: ['LLM-based semantic search', 'Corpus: unstructured customer comments', 'Identifies coil-positioning friction', 'Extracts R&D “pains” at scale', 'Structured output for product teams'],
    mine: true,
  },
  {
    n: 3,
    name: 'VR Training Simulation',
    points: ['Unity engine + Meta Quest', 'Fed from digital-twin logic', 'Technician coil-change practice', 'No live scanner needed', 'Reduces training cost & time'],
    mine: false,
  },
] as const;

export const ARCHITECTURE = {
  specs: 'd_model 128 · 3 layers · 4 heads · FF 512',
  loss: 'Loss = token_CE + 0.3 × duration_MSE (log-space)',
  tiers: [
    {
      n: 1,
      name: 'Conditioning Encoder',
      role: 'Who is on the table, when, with which coils',
      details: ['Patient features ×5 — age · weight · height · PTAB · direction', 'Temporal features ×5 — hour · DOW · week, sin/cos encoded', 'Coil features ×30 — BC · SP1–8 · HE1–4 · body-region embedding', 'Conditioning scale buffer — stops LayerNorm erasing categorical signal'],
      plain: 'Turns patient, time and coil context into memory vectors the rest of the model attends to.',
    },
    {
      n: 2,
      name: 'Token Decoder',
      role: 'What happens next',
      details: ['Autoregressive, causal self-attention', 'Cross-attention into Tier-1 memory', 'Vocabulary: 18 source IDs (MRI_CCS · EXU · MSR · MPT · PAD)', 'softmax · top-k sampling'],
      plain: 'Writes the day as a sequence of hardware events, one token at a time — like a language model for scanner logs.',
    },
    {
      n: 3,
      name: 'Duration Head',
      role: 'How long it takes — with honesty about spread',
      details: ['Single-pass bidirectional encoder', 'Key-padding masks hide PAD + END', 'log1p transform, duration_scale = 60 s', 'Output per token: μ ± σ (seconds)'],
      plain: 'Reads the whole generated sequence and prices every event in seconds — mean and uncertainty.',
    },
  ],
} as const;

export const ALTERNATION = {
  phases: [
    { model: 'exchange', label: 'Exchange', sub: 'Startup', detail: 'START → body₁' },
    { model: 'exam', label: 'Examination', sub: 'Scan sequence', detail: 'body₁ scans' },
    { model: 'exchange', label: 'Exchange', sub: 'Between', detail: 'body₁ → body₂' },
    { model: 'exam', label: 'Examination', sub: 'Scan sequence', detail: 'body₂ scans' },
    { model: 'exchange', label: 'Exchange', sub: 'Shutdown', detail: 'END' },
  ],
  handoffs: 'Handoff tokens: T:100 opens a scan block · T:104 closes it',
  conditioning: ['Age · Weight · Height · Direction', 'Body region(s)', 'Coil state · Serial ID', 'Sequence type · PTAB'],
  output: 'Per-token durations with uncertainty → μ ± σ (log-space, duration_scale = 60 s)',
} as const;

export const WHITEBOX = {
  black: ['Single neural network end-to-end', 'No explanation of predictions', 'No uncertainty bounds', 'Cannot isolate Exchange vs Examination logic', 'Fails to adapt when coil configuration changes', 'Point estimate only — no schedule-slack guidance'],
  white: ['Three explicit tiers: Conditioning → Decoder → Duration', 'Every prediction includes μ ± σ', 'Conditioning features are named and interpretable', 'Exchange / Examination independently testable', 'Conditioning scale buffer prevents feature erasure', 'Clinicians and engineers can understand every output'],
} as const;

export const DECISIONS = [
  { decision: 'Log-space durations', why: 'MRI scan times are right-skewed — Gaussian fails', evidence: 'Mean ≫ median without log1p transform' },
  { decision: 'Conditioning scale buffer', why: 'LayerNorm crushes small categorical embeddings next to raw features (Age ≈ 50)', evidence: 'Training diverged without buffer — stable with it' },
  { decision: 'Separate Exchange / Exam models', why: 'Phase semantics differ fundamentally — a combined model is ambiguous', evidence: 'Independent models generalise better' },
  { decision: 'Orchestrator model', why: 'Ground-truth patient sequences unavailable at inference time', evidence: 'Generates body-region order autonomously' },
  { decision: 'Bidirectional duration head', why: 'Duration of token i depends on neighbours — unidirectional underestimates', evidence: 'Single-pass encoder sees full context via key-padding mask' },
] as const;

/* ---------------------------------------------------------------- results */

export const HEADLINE = [
  { value: '3', unit: 'models', label: 'generative models live', sub: 'on 40 real MRI scanners' },
  { value: '±12%', unit: '', label: 'day-length fidelity', sub: 'median across 40 scanners' },
  { value: '<1.8', unit: 'min', label: 'duration MAE', sub: 'σ calibrated within 20%' },
  { value: '~$17.7K', unit: '/yr', label: 'recoverable per machine', sub: '71% of quantified loss' },
  { value: '88.3%', unit: '', label: 'multi-agent diagnosis', sub: 'master thesis · Siemens data' },
] as const;

export const DAY_FIDELITY = {
  regionOrder: '>78%',
  regionOrderSub: 'of days match ground-truth body-region sequence (validation set)',
  dayLength: '±12%',
  dayLengthSub: 'simulated vs real day length (median across 40 scanners)',
  exchangeError: '<15 s',
  exchangeErrorSub: 'median error on patient-transition durations',
} as const;

/** Gantt lanes: minutes from 08:00 (240 = 12:00). Derived from the deck figure. */
export const GANTT = {
  start: '08:00',
  ticks: ['08:00', '09:00', '10:00', '11:00', '12:00'],
  total: 240,
  real: [
    { region: 'HEAD', from: 0, to: 50 },
    { region: 'EXCH', from: 50, to: 60 },
    { region: 'SPINE', from: 60, to: 125 },
    { region: 'EXCH', from: 125, to: 134 },
    { region: 'PELVIS', from: 134, to: 187 },
    { region: 'EXCH', from: 187, to: 194 },
    { region: 'ABD', from: 194, to: 240 },
  ],
  sim: [
    { region: 'HEAD', from: 0, to: 55 },
    { region: 'EXCH', from: 55, to: 62 },
    { region: 'SPINE', from: 62, to: 130 },
    { region: 'EXCH', from: 130, to: 139 },
    { region: 'PELVIS', from: 139, to: 190 },
    { region: 'EXCH', from: 190, to: 197 },
    { region: 'ABD', from: 197, to: 240 },
  ],
} as const;

export const DURATION_PRED = {
  regions: ['HEAD', 'NECK', 'SPINE', 'PELVIS', 'ABD'],
  observed: [9.0, 7.0, 22.0, 16.0, 18.0],
  predicted: [9.2, 7.4, 21.8, 15.8, 18.5],
  metrics: [
    ['Mean abs. error', '< 1.8 min'],
    ['Median abs. error', '< 1.1 min'],
    ['σ calibration', 'within 20%'],
    ['Token sequence acc.', '> 82%'],
  ],
  interpretation:
    'The model predicts not just the mean duration but the spread — enabling principled schedule-slack allocation.',
} as const;

export const ORCHESTRATOR = {
  truth: ['HEAD', 'NECK', 'BREAK', 'SPINE', 'SPINE', 'PELVIS'],
  predicted: ['HEAD', 'NECK', 'BREAK', 'SPINE', 'ABD', 'PELVIS'],
  mismatchIndex: 4,
  exampleNote: '5/6 tokens correct — edit distance 1',
  editDistance: '1.4 tokens/day',
  breakAccuracy: '89%',
  validation:
    'Hold out the last 2 weeks per scanner; compare predicted body-region sequence to ground truth by Levenshtein distance on region tokens.',
  mustLearn: ['Sequence of body regions for a new day, from scanner history', 'BREAK tokens (lunch, handover gaps)', 'Clinical patterns — HEAD/NECK cluster in the morning', 'Per-scanner seasonal patterns'],
} as const;

export const UNCERTAINTY = {
  head: { mu: 8.0, sigma: 1.2 },
  spine: { mu: 18.0, sigma: 5.0 },
  note: 'Durations predicted in log-space (log1p), then exponentiated — handles the right-skew inherent to MRI scan times (30 s – 45 min).',
} as const;

/**
 * Provenance: June 2026 staged validation runs. Before-states and gates are
 * confirmed in repo code (validate_stage1_examination.py hardcodes the broken
 * 26-row baseline; examination_duration_calibration.py documents the flat
 * ~49 s prior; step 05 carries the entropy >0.7 gate and region-collapse
 * flag). The after-numbers come from the run outputs on Databricks DBFS.
 */
export const IMPROVEMENT_PROVENANCE =
  'validated Jun 2026 · harnesses in DatabricksPipeline/csv_pipeline · run artifacts on Databricks';

export const IMPROVEMENT = [
  {
    stage: 'Stage 1 · 2026-06-04',
    title: 'Examination collapse fixed',
    before: '26 rows',
    after: '21,620 rows',
    beforeNum: 26,
    afterNum: 21_620,
    detail: 'Generated exam events per run — model no longer terminates after a single token.',
  },
  {
    stage: 'Stage 2 · 2026-06-11',
    title: 'Region realism restored',
    before: 'phantom regions ~91%',
    after: 'entropy 0.808',
    detail: 'Phantom body regions (never in real schedules) eliminated; generated mix matches clinical patterns.',
  },
  {
    stage: 'Stage 3 · 2026-06-13',
    title: 'Durations calibrated',
    before: 'flat ~50 s for every type',
    after: '1.7 min ≈ real 1.75 min',
    detail: 'Per-type medians land on real values (scout 17 s vs real 19 s); spread restored.',
  },
] as const;

export const MRRT = {
  challenge: ['Thousands of unstructured customer comments in the MRRT database', 'Manual triage of R&D pains is slow and subjective', 'Key signals buried in free text', 'No systematic ranking by frequency or impact'],
  exampleIn: '“Had to manually override coil detection three times during the spine exam. Very frustrating for the technician.”',
  exampleOut: [
    ['type', 'friction'],
    ['component', 'coil_detection'],
    ['body_region', 'SPINE'],
    ['frequency', 'high'],
  ],
  engine: 'embedding + retrieval + LLM',
  findings: [
    'Coil-detection failures are the #1 technician friction point in SPINE examinations — surfaced from hundreds of comments',
    'Manual override rate is 3× higher for spine coils than head coils',
    'Signal missed entirely by structured survey data',
    'Agent quantifies frequency automatically — R&D previously had none',
  ],
  pains: [
    { cat: 'Coil detection', count: 312, impact: 'High' },
    { cat: 'Workflow interrupts', count: 204, impact: 'High' },
    { cat: 'Table positioning', count: 178, impact: 'Medium' },
    { cat: 'Patient comfort', count: 145, impact: 'Medium' },
    { cat: 'Image quality', count: 97, impact: 'High' },
    { cat: 'Scan abort rate', count: 89, impact: 'Medium' },
  ],
  caveat: 'Example output — actual counts vary by corpus',
  impact:
    'Directly informs coil-firmware priority and positions coil reliability as a selling point in new MRI configurations.',
} as const;

/* ----------------------------------------------------------------- thesis */

export const THESIS = {
  name: 'Agentic Infra Co-Pilot',
  framing: 'Master thesis — multi-agent fault diagnosis for MRI infrastructure',
  pitch:
    'Three specialist LLM agents — Governance, Hardware, Telemetry — each with its own retrieval store, collaborate under a structured autonomy protocol; a synthesizer fuses their findings into one diagnosis.',
  question: 'Does agent collaboration produce diagnostic ability no single agent achieves alone?',
  siemensData: ['~50 real Siemens Healthineers PDFs', 'MAGNETOM (Vida / Sola) operator manuals', 'DICOM conformance statements', 'Safety & SOP documents', 'Context: ~40 customer installations'],
  agents: [
    { name: 'Governance', scope: 'SLA & policy', docs: 418 },
    { name: 'Hardware', scope: 'MRI & DICOM', docs: 9_853 },
    { name: 'Telemetry', scope: 'Event logs & safety', docs: 945 },
  ],
  protocol: ['ANSWER', 'PARTIAL', 'CONSULT', 'REDIRECT', 'CLARIFY', 'REFUSE'],
  stack: 'DSPy · FastAPI micro-services · ChromaDB RAG per domain · parallel orchestration',
  llmTiers: 'Claude Haiku 4.5 (reasoners) · Claude Sonnet 4.6 (synthesizer) · GPT-4.1 (router)',
  results: {
    modes: [
      { mode: 'Governance only', acc: 50.0, kind: 'single' },
      { mode: 'Telemetry only', acc: 55.0, kind: 'single' },
      { mode: 'Hardware only', acc: 71.7, kind: 'single' },
      { mode: 'Single agent + all data', acc: 83.3, kind: 'baseline' },
      { mode: 'Full multi-agent', acc: 88.3, kind: 'mas' },
    ],
    judgeBaseline: 0.855,
    judgeMas: 0.942,
    emergence: '4 / 12 cases',
    marginPp: 5.0,
    latencyMas: '~127 s',
    latencyBaseline: '~59 s',
    setup: '60 evaluations · 12 fault cases × 5 modes',
  },
  demo: ['React chat UI with live delegation badges', 'Full reasoning-chain trace per diagnosis', 'Specialist panels + sources + risk badges', 'One-command docker-compose bring-up'],
  /** Thesis abstract, verbatim framing of the Telekom case study. */
  crossDomain: {
    headline: 'Telekom IoT: +6 pp',
    sub: 'MAS judge gain · +33 pp semantic similarity · preliminary cross-domain case study (thesis)',
  },
} as const;

/**
 * Related ablation from the earlier T-Labs study (iot-ipe-copilot deck,
 * Run 4, 2026-04-07) — DIFFERENT domain and data (Deutsche Telekom IoT-IPE),
 * shown in the appendix for methodology breadth, never mixed into the
 * Siemens MRI numbers.
 */
export const TELEKOM_STUDY = {
  title: 'Related ablation — IoT-IPE Copilot (Deutsche Telekom / T-Labs data)',
  date: 'Run 4 · 2026-04-07 · 9-mode Pareto sweep',
  head: ['Mode', 'Keyword', 'Judge', '$ / query', 'Latency', 'Verdict'],
  rows: [
    ['discovery_coding (2 agents)', '96%', '0.90', '$0.0352', '49.5 s', 'Pareto-optimal'],
    ['mas_full_concat (3, concat merge)', '95%', '0.85', '$0.0522', '10.6 s', 'fastest'],
    ['mas_full (3, LLM synthesis)', '81%', '0.82', '$0.0778', '127.1 s', 'dominated'],
  ],
  lesson:
    '“Two specialists with naïve concatenation beat three with LLM synthesis — the cause is in the merging step, not the headcount.”',
  note: 'Different domain, data and date than the Siemens MRI evaluation — the thesis cites it as a preliminary cross-domain replication (MAS +6 pp judge, +33 pp semantic similarity).',
} as const;

/* -------------------------------------------------------------- ops/cloud */

export const OPS = {
  local: {
    title: 'Local execution (CPU)',
    rows: [
      ['Entry point', 'run_all.py'],
      ['Steps 1 → 7', 'Preprocess → Train → Simulate → Visualise'],
      ['Framework', 'PyTorch (d_model 128 · 3 layers · 4 heads)'],
      ['Duration mode', 'log (log1p, scale 60 s)'],
      ['Output', 'Timestamped CSV + HTML Gantt charts'],
    ],
  },
  cloud: {
    title: 'Cloud execution (Databricks)',
    rows: [
      ['Entry point', 'spark_pipeline.py'],
      ['Method', 'applyInPandas — parallel preprocessing per scanner'],
      ['Advantage', 'All 40 scanners preprocessed concurrently'],
      ['Artefacts', 'Same model weights loaded identically'],
      ['Output', 'Parquet + shared object storage'],
    ],
  },
  parity:
    'Both environments produce identical artefacts — examination_model_best.pt · exchange_model_best.pt · orchestration_model_best.pt',
  status: [
    { step: '1 · 1b', label: 'Preprocessing (Exchange · Orchestration)', state: 'done' },
    { step: '2 · 3 · 2c', label: 'Train Exchange · Examination · Orchestration', state: 'done' },
    { step: '4 · 4b', label: 'Simulated + autonomous day generation', state: 'refining' },
    { step: '5', label: 'Bucket generation · per-customer fine-tuning · MRRT integration', state: 'planned' },
  ],
} as const;

export const CLOUD_READINESS = [
  { item: 'Local ↔ cloud artefact parity', state: 'ready', note: 'same weights, both worlds' },
  { item: 'Parallel preprocessing at fleet scale', state: 'ready', note: 'applyInPandas, 40 scanners concurrently' },
  { item: 'Shared storage for downstream models', state: 'ready', note: 'Parquet + object storage' },
  { item: 'Per-step timing instrumentation', state: 'ready', note: 'pipeline live on Databricks' },
  { item: 'Production hardening', state: 'progress', note: 'checkpointing + per-step result caching' },
  { item: 'Bucket pre-generation', state: 'planned', note: '1,000 candidates/bucket → instant day assembly' },
] as const;

export const SCALING = {
  current: ['One set of weights per scanner serial', 'All 40 scanners share a population-level backbone', 'Serial embedding distinguishes customers', 'Strong on known scanners'],
  target: ['Shared backbone pre-trained across all sites', 'Site-specific adapter heads — fine-tuned on ≤ 4 weeks of local data', 'Customer-ID embedding as conditioning', 'Federated or centralised — data can stay on-premises', 'Cold start: < 2 h on CPU with ~500 days of history'],
  insight:
    'The conditioning architecture was designed for this: customer ID slots into the encoder without architectural change.',
} as const;

/* ---------------------------------------------------------------- roadmap */

export const ROADMAP = [
  { q: 'Q3 2026', now: true, title: 'Model Stabilisation', items: ['Exchange · Examination · Orchestrator trained', 'Orchestrator validation', 'Bucket-based generation', 'Databricks production'] },
  { q: 'Q4 2026', now: false, title: 'Customer Intelligence', items: ['Per-customer fine-tuning', 'MRRT Insight Agent MVP', 'ROI simulation module', 'Coil upgrade modelling', 'Multi-site data ingestion', 'Predictive maintenance module'] },
  { q: 'Q1 2027', now: false, title: 'Integration & Validation', items: ['VR training integration', 'ROI dashboard', 'Clinical validation study', 'External pilot site', 'Real-time schedule API'] },
  { q: 'Q2 2027', now: false, title: 'Scale & Deploy', items: ['Multi-site deployment', 'SaaS product packaging', 'Sales tool integration', 'Coil portfolio simulator', 'Customer portal launch'] },
] as const;

export const NEXT_STEPS = [
  { n: 1, title: 'Finalize orchestrator validation', gate: 'gate: mean edit distance ≤ 2.0', desc: 'Hold-out evaluation across all 40 scanners; edit-distance distribution + BREAK-token F1.' },
  { n: 2, title: 'Bucket-based generation', gate: '1,000 candidates / bucket', desc: 'Pre-generate Exchange + Examination candidates — instant day assembly, no model calls at query time.' },
  { n: 3, title: 'Databricks production hardening', gate: 'checkpoint + cache', desc: 'Resolve pipe-close edge cases; per-step result caching for resilience.' },
  { n: 4, title: 'Per-customer fine-tuning pipeline', gate: 'customer → weights registry', desc: 'Shared backbone + site fine-tune at reduced LR; customer-ID embedding.' },
  { n: 5, title: 'MRRT agent integration', gate: 'first SPINE-coil pain report', desc: 'Connect retrieval agent to the comment corpus; structured output schema.' },
  { n: 6, title: 'External pilot site', gate: 'simulated vs actual day', desc: 'First external MRI site: live ingestion, end-to-end run, side-by-side comparison.' },
] as const;

/* ---------------------------------------------------------------- finance */

export const ROI = {
  baseline: 25.0,
  steps: [
    { label: 'Downtime', sub: 'predictive maintenance', delta: -7.0 },
    { label: 'Throughput', sub: '+14.5% utilisation', delta: -6.2 },
    { label: 'Wait reduction', sub: '−44.8% patient wait', delta: -4.5 },
  ],
  residual: 7.3,
  recoverable: '~$17,700',
  recoverablePct: '71%',
  perMachineK: 17.7,
  fleetDefault: 40,
  disclaimer:
    'Estimates based on Silva-Aravena et al. (2025) and industry downtime analysis. Actual ROI varies by site.',
} as const;

export const BUSINESS_CASES = [
  { n: 1, title: 'Coil-upgrade ROI simulation', use: 'Sales must justify a new coil to a customer.', how: 'Run the twin with the new coil conditioning; compare simulated throughput before/after.', result: 'Quantified ROI before purchase — faster sales cycle, lower buyer risk.' },
  { n: 2, title: 'Predictive maintenance', use: 'Tube replacement is reactive — ≈ $15K per unplanned downtime event.', how: 'Monitor tube life + arcing rate; twin flags degradation 2 weeks ahead.', result: 'Planned service replaces crisis — downtime scheduled at low utilisation.' },
  { n: 3, title: 'VR technician training', use: 'Training on a live scanner costs ≈ $500/hour in lost revenue.', how: 'Feed the twin’s schedule into Unity / Meta Quest coil-change practice.', result: '80% of competency hours moved off the live scanner, available 24/7.' },
] as const;

export const SUMMARY = {
  remember: [
    { title: 'We have a working generative twin.', body: 'Exchange, Examination and Orchestration models trained and validated on 40 real MRI scanners — synthetic days produced end-to-end today.' },
    { title: 'The architecture is designed to scale.', body: 'White-box μ ± σ predictions, interpretable conditioning and modular design extend to new sites and use cases without rebuilding.' },
    { title: 'The ROI is concrete and quantifiable.', body: '≈ $17,700 per machine per year in recoverable value — before coil simulation and VR training upside.' },
  ],
  ask: [
    { title: 'Pilot site', desc: 'Agree the first external site for live data ingestion' },
    { title: 'ROI protocol', desc: 'Define how recovered value is measured on site' },
    { title: 'Technical walkthrough', desc: 'Schedule the deep-dive with the engineering team' },
    { title: 'Data agreement', desc: 'Confirm sharing terms for external scanner data' },
  ],
} as const;

/* ---------------------------------------------------------------- journey */

export const JOURNEY = [
  { phase: 'Foundation', year: '2024', desc: 'Event-log corpus from 40 scanners; 18-token vocabulary, 11 body regions, 15 sequence types.', outcome: 'A modelling-ready view of how MRI days actually run.' },
  { phase: 'Architecture', year: '2025', desc: 'White-box three-tier transformer; alternating Exchange / Examination design; μ ± σ everywhere.', outcome: 'Interpretable by construction — not post-hoc.' },
  { phase: 'Validated models', year: '2026', desc: 'Three models trained, validated and calibrated against held-out days.', outcome: '±12% day fidelity · <1.8 min MAE.' },
  { phase: 'Cloud & scale', year: '2026', desc: 'Databricks pipeline with artefact parity; multi-site adapter architecture designed.', outcome: 'Fleet-scale is a scale-up, not a rewrite.' },
] as const;

export const CONTRIBUTIONS = [
  { title: 'Alternating Pipeline models', desc: 'Exchange, Examination & Orchestration transformers; μ ± σ duration heads; validation harnesses.', mine: true },
  { title: 'Databricks / cloud pipeline', desc: 'spark_pipeline.py parallelisation, artefact parity, provenance & freshness tooling.', mine: true },
  { title: 'MRRT Insight Agent', desc: 'LLM semantic retrieval over customer comments; structured pain extraction.', mine: true },
  { title: 'Agentic Infra Co-Pilot', desc: 'Master thesis — multi-agent MRI fault diagnosis on Siemens data.', mine: true, thesis: true },
  { title: 'VR training simulation', desc: 'Unity / Meta Quest technician training fed from the twin.', mine: false },
] as const;

export const RUNBOOK = [
  ['1', 'Preprocess Exchange sequences', 'Exchange CSV per scanner, duration priors'],
  ['1b', 'Preprocess Orchestration data', 'Day-level body-region sequences with BREAK tokens'],
  ['2', 'Train Exchange model', 'exchange_model_best.pt'],
  ['3', 'Train Examination model', 'examination_model_best.pt'],
  ['2c', 'Train Orchestration model', 'orchestration_model_best.pt'],
  ['4', 'Simulate days (ground-truth patients)', 'simulated_day_*.csv + Gantt HTML'],
  ['4b', 'Orchestrated simulation (autonomous)', 'orchestrated_day_*.csv'],
  ['5', 'Visualisations (exchange + examination)', 'Duration distributions, body-region analysis'],
  ['5b', 'Orchestration visualisations', 'Comparison charts, BREAK timelines'],
  ['6', 'Per-customer simulation', 'Customer-specific output directories'],
  ['7', 'General visualisations', 'Summary dashboard HTML'],
] as const;
