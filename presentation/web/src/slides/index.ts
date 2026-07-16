import type { SlideDef } from '../deck/types';
import { S01Title } from './s01Title';
import { S02Numbers } from './s02Numbers';
import { S03Agenda } from './s03Agenda';
import { S04FourTwins } from './s04FourTwins';
import { S05Problem } from './s05Problem';
import { S06HiddenCost } from './s06HiddenCost';
import { S07Pivot } from './s07Pivot';
import { S08Gap } from './s08Gap';
import { S09Journey } from './s09Journey';
import { S10Overview } from './s10Overview';
import { S11Workstreams } from './s11Workstreams';
import { S12Data } from './s12Data';
import { S13Contributions } from './s13Contributions';
import { S14Alternation } from './s14Alternation';
import { S15Architecture } from './s15Architecture';
import { S16WhiteBox } from './s16WhiteBox';
import { S17Uncertainty } from './s17Uncertainty';
import { S18Decisions } from './s18Decisions';
import { S19Gantt } from './s19Gantt';
import { S20Fidelity } from './s20Fidelity';
import { S21Duration } from './s21Duration';
import { S22Orchestrator } from './s22Orchestrator';
import { S23Improvement } from './s23Improvement';
import { S24Mrrt } from './s24Mrrt';
import { S25ThesisIntro } from './s25ThesisIntro';
import { S26ThesisArch } from './s26ThesisArch';
import { S27ThesisResults } from './s27ThesisResults';
import { S28ThesisDemo } from './s28ThesisDemo';
import { S29Ops } from './s29Ops';
import { S30Cloud } from './s30Cloud';
import { S31Scaling } from './s31Scaling';
import { S32Roadmap } from './s32Roadmap';
import { S33NextSteps } from './s33NextSteps';
import { S34Roi } from './s34Roi';
import { S35Cases } from './s35Cases';
import { S36Summary } from './s36Summary';
import { S37Ask } from './s37Ask';
import { S38Thanks } from './s38Thanks';
import { A01Transformer } from './a01Transformer';
import { A02Runbook } from './a02Runbook';
import { A03ModelRef } from './a03ModelRef';
import { A04MrrtMethod } from './a04MrrtMethod';
import { A05ThesisEval } from './a05ThesisEval';
import { A06Sources } from './a06Sources';

export const SLIDES: SlideDef[] = [
  {
    id: 's01', act: 0, chrome: 'hero', element: S01Title, title: 'MRI Digital Twin — title',
    notes: 'Welcome. Frame the hour: a results report on three years of the MRI Digital Twin, plus my thesis work on the same data estate. 45 minutes of content, 15 for discussion.',
  },
  {
    id: 's02', act: 0, element: S02Numbers, title: 'Three years in five numbers',
    notes: 'The whole talk on one slide. Read the five numbers slowly, left to right. Promise: every claim today is measured, and each of these gets its own act. Do not explain them yet.',
  },
  {
    id: 's03', act: 0, element: S03Agenda, title: 'How the next hour runs',
    notes: 'Ten seconds only. Point out the schedule bar at the top — it mirrors this agenda for the whole talk. Then move.',
  },
  {
    id: 's04', act: 1, element: S04FourTwins, title: 'The Four Twins ecosystem',
    notes: 'Strategic anchor: Siemens runs four twin pillars; ours is the Customer Twin — workflow simulation and schedule optimization. Mandated, not a hobby project.',
  },
  {
    id: 's05', act: 1, element: S05Problem, title: 'The scheduling problem',
    notes: 'The core tension: 30 seconds to 45 minutes on the same scanner. One overrun cascades. Land the root-cause line: static scheduling treats a stochastic process as deterministic.',
  },
  {
    id: 's06', act: 1, element: S06HiddenCost, title: 'The hidden cost',
    notes: '$25K per machine per year is the stake. The literature numbers (−44.8% wait, +14.5% utilization) prove digital twins recover this class of loss — cite Silva-Aravena 2025 verbally.',
  },
  {
    id: 's07', act: 1, element: S07Pivot, title: 'From reporting to simulation',
    notes: 'The pivot slide: today static reporting, tomorrow generative simulation with uncertainty bounds. This is the mandate we were given three years ago.',
  },
  {
    id: 's08', act: 1, element: S08Gap, title: 'Six gaps — six answers',
    notes: 'Scan the third column, not every row. Close the act with the so-what: the gap was a generative, uncertainty-aware twin, and we built it.',
  },
  {
    id: 's09', act: 2, fragments: 3, element: S09Journey, title: 'The three-year journey',
    notes: 'Step through the four phases (arrow key each): 2024 foundation, 2025 architecture, 2026 validation and cloud. Say it plainly: many model variants were tried, evaluated and retired along the way — the version history is the audit trail.',
  },
  {
    id: 's10', act: 2, element: S10Overview, title: 'Customer Twin system overview',
    notes: 'The original system diagram from the handover. Walk it left to right: five data inputs, three transformer models with μ±σ, five output products. Emphasize white-box.',
  },
  {
    id: 's11', act: 2, element: S11Workstreams, title: 'Three parallel workstreams',
    notes: 'One data foundation, three products: the generative pipeline, the MRRT semantic agent, and VR training. All share the twin engine.',
  },
  {
    id: 's12', act: 2, element: S12Data, title: 'The data foundation',
    notes: '40 real scanners, two months of event logs, structured into an 18-token vocabulary. This corpus is the ground truth everything else is measured against.',
  },
  {
    id: 's13', act: 2, element: S13Contributions, title: 'Where my work sits',
    notes: 'Ownership, stated plainly: the three pipeline models, the cloud pipeline, the MRRT agent, and the thesis system are my builds; VR is the team’s. Team of 4 — Luke, Navneet, Martina, Georg (team lead). Sebastian provided the MRRT corpus and feedback.',
  },
  {
    id: 's14', act: 3, element: S14Alternation, title: 'The alternating pipeline',
    notes: 'How a day gets generated: Exchange and Examination models alternate, handing off via tokens, conditioned on patient, coils, time, and serial. Alternation IS the architecture.',
  },
  {
    id: 's15', act: 3, fragments: 3, element: S15Architecture, title: 'Inside the model',
    notes: 'Step through the three tiers with plain-language jobs: who is on the table → what happens next → how long it takes. Fourth step reveals the loss line. Full diagram is appendix A1 if pressed.',
  },
  {
    id: 's16', act: 3, element: S16WhiteBox, title: 'White-box by construction',
    notes: 'Positioning against black-box competitors: named features, independent testability, μ±σ everywhere. In clinical settings, explainability is the product.',
  },
  {
    id: 's17', act: 3, element: S17Uncertainty, title: 'Uncertainty that means something',
    notes: 'Live moment: drag the slack slider. HEAD gets tight slots, SPINE gets wide buffers — same policy, buffer where variance lives. This is what σ buys operationally.',
  },
  {
    id: 's18', act: 3, element: S18Decisions, title: 'Hard-won architecture decisions',
    notes: 'Five decisions, each validated by a failure we diagnosed. If the audience is technical, the conditioning-scale-buffer row is the best story (LayerNorm erasing categorical signal).',
  },
  {
    id: 's19', act: 4, element: S19Gantt, title: 'A simulated day vs reality',
    notes: 'The signature visual. Hit "replay generation" and let the scan line assemble the simulated day under the real one. Same scanner, same date. Pause here.',
  },
  {
    id: 's20', act: 4, element: S20Fidelity, title: 'Day-level fidelity',
    notes: 'Three held-out validation numbers: >78% region order, ±12% day length across 40 scanners, <15s exchange error. Good enough to schedule against.',
  },
  {
    id: 's21', act: 4, element: S21Duration, title: 'Duration prediction',
    notes: 'Bars nearly overlap per region — that is the point. MAE under 1.8 minutes, and σ itself is calibrated within 20%: the error bars are trustworthy.',
  },
  {
    id: 's22', act: 4, element: S22Orchestrator, title: 'Orchestrator validation',
    notes: 'The orchestrator writes the day autonomously — no patient list exists at inference. Mean edit distance 1.4 tokens/day, break placement 89%.',
  },
  {
    id: 's23', act: 4, fragments: 2, element: S23Improvement, title: 'How the model improved',
    notes: 'Honesty slide, step through three stages: collapse fixed (26→21,620 rows), phantom regions eliminated, durations calibrated to real means. Numbers are from the June 2026 staged validation runs; harnesses live in DatabricksPipeline/csv_pipeline, and if challenged the primary run artifacts are on Databricks DBFS.',
  },
  {
    id: 's24', act: 4, element: S24Mrrt, title: 'MRRT Insight Agent — findings',
    notes: 'Free text in, structured pain records out. Headline finding: coil detection is the #1 spine friction, override rate 3× head coils — invisible to surveys. Counts are example output.',
  },
  {
    id: 's25', act: 5, element: S25ThesisIntro, title: 'Agentic Infra Co-Pilot — what it is',
    notes: 'Switch hats: master thesis, same Siemens data estate. Three specialist agents diagnose MRI infrastructure faults. The research question: does collaboration beat any single agent? Credit Sebastian verbally — data access and feedback.',
  },
  {
    id: 's26', act: 5, element: S26ThesisArch, title: 'Three specialists, one diagnosis',
    notes: 'Governance, Hardware (9,853 docs), Telemetry — each with its own retrieval store, talking through a six-verb autonomy protocol, fused by a synthesizer. Refusing is a feature.',
  },
  {
    id: 's27', act: 5, element: S27ThesisResults, title: 'Measured emergence',
    notes: 'The money chart: full MAS 88.3% vs 83.3% for one agent with ALL the data — +5.0pp is pure collaboration. Judge score 0.942. Be honest about latency: 127s vs 59s. Numbers confirmed against thesis Chapter 5. Fourth chip: the thesis cites the Telekom IoT study as a preliminary cross-domain replication (+6 pp judge) — full Pareto detail on A5 if asked.',
  },
  {
    id: 's28', act: 5, element: S28ThesisDemo, title: 'What we can show live',
    notes: 'Demo-ability: React chat UI with live delegation badges and reasoning traces; docker-compose bring-up. Offer a live session or the screenshot. Close the act: twin predicts, copilot explains.',
  },
  {
    id: 's29', act: 6, element: S29Ops, title: 'Local ↔ Databricks parity',
    notes: 'Operations today: identical artefacts from laptop CPU and Databricks. Status strip: preprocessing and training done, generation refining, bucket stage planned.',
  },
  {
    id: 's30', act: 6, element: S30Cloud, title: 'Cloud potential',
    notes: 'Readiness board: four green rows already true today, hardening in progress, bucket pre-generation planned. Cloud is a scale-up, not a rewrite — the weights already run there.',
  },
  {
    id: 's31', act: 6, element: S31Scaling, title: 'Scaling to multi-site',
    notes: 'The fleet plan: shared backbone plus site adapter heads fine-tuned on ≤4 weeks of local data; cold start under 2 hours on CPU. Federated option keeps data on premises.',
  },
  {
    id: 's32', act: 7, element: S32Roadmap, title: 'Development roadmap',
    notes: 'Q3 committed (validation, buckets, hardening), Q4 customer intelligence, Q1 pilot + clinical validation, Q2 scale and SaaS packaging.',
  },
  {
    id: 's33', act: 7, element: S33NextSteps, title: 'Immediate next steps',
    notes: 'Six steps, each with a measurable exit gate — orchestrator gate is mean edit distance ≤2.0. Nothing open-ended.',
  },
  {
    id: 's34', act: 7, fragments: 4, element: S34Roi, title: 'Financial impact per machine',
    notes: 'Build the waterfall step by step (arrow key each): $25K baseline, −$7K downtime, −$6.2K throughput, −$4.5K wait → $7.3K residual. ~$17.7K recoverable, 71%. The final step also reveals the fleet slider — drag to 40 machines: ≈$708K/yr. Keep the estimates disclaimer visible.',
  },
  {
    id: 's35', act: 7, element: S35Cases, title: 'Three business cases',
    notes: 'Coil ROI before purchase, predictive maintenance instead of $15K crisis events, VR training off the live scanner. Same twin, different conditioning — no new models.',
  },
  {
    id: 's36', act: 7, element: S36Summary, title: 'Three things to remember',
    notes: 'Slow down. One: working generative twin on 40 real scanners. Two: architecture built to scale. Three: quantified ROI. ',
  },
  {
    id: 's37', act: 7, element: S37Ask, title: 'The ask',
    notes: 'The decision slide: pilot site and data agreement need a decision today; ROI protocol and the technical walkthrough follow by calendar.',
  },
  {
    id: 's38', act: 7, chrome: 'hero', element: S38Thanks, title: 'Thank you / Q&A',
    notes: 'Open the floor. Appendix lives behind this slide: full architecture (A1), runbook (A2), model reference (A3), MRRT method (A4), thesis eval design (A5), sources & open items (A6).',
  },
  {
    id: 'a01', act: 8, appendix: true, element: A01Transformer, title: 'Full transformer architecture',
    notes: 'Unabridged three-tier diagram from the handover deck — use for deep architecture questions.',
  },
  {
    id: 'a02', act: 8, appendix: true, element: A02Runbook, title: 'run_all.py runbook',
    notes: 'The 11-step local pipeline; Databricks runs the same stages via spark_pipeline.py.',
  },
  {
    id: 'a03', act: 8, appendix: true, element: A03ModelRef, title: 'Model configuration reference',
    notes: 'Per-tier feature groups, masks, and the loss/spec line — for parameter-level questions.',
  },
  {
    id: 'a04', act: 8, appendix: true, element: A04MrrtMethod, title: 'MRRT method detail',
    notes: 'Retrieval + LLM extraction pipeline and the structured output schema.',
  },
  {
    id: 'a05', act: 8, appendix: true, element: A05ThesisEval, title: 'Thesis evaluation design',
    notes: 'RQ1–RQ3 with answers, 12 cases × 5 modes, and the results-provenance warning (final JSON vs stale README).',
  },
  {
    id: 'a06', act: 8, appendix: true, element: A06Sources, title: 'Sources & open items',
    notes: 'Every source used in the deck plus the consolidated TODO list to resolve before presenting.',
  },
];
