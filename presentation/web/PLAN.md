# MRI Digital Twin — 60-min Interactive Web Presentation (React + Vite + Tailwind)

## Context

Luke reports to his boss's boss (senior AI/ML engineering leader): ~3 years of the **MRI Digital Twin / Customer Twin** programme for Siemens Healthineers + his master-thesis **agentic-infra-copilot** (multi-agent MRI fault diagnosis, also on Siemens data). Results and outcomes first, Duarte-style narrative (tension → insight → impact → direction), each act ends with a "so what / decision" line. Deliverable: a runnable interactive web deck (not PDF), Siemens-Healthineers clinical-grade design, plus a missing-info list.

**Sources of truth:** `presentation/main.tex` + `presentation/svgs/*.svg` (all facts extracted — see Fact Inventory below); `agentic-infra-copilot` repo for the thesis section (authoritative results: `results/ablation/thesis_summary.json`, NOT the stale README table). Anything else → visible `TODO` chip + Missing-Info list. Scaffolding inspiration: `iot-ipe-copilot/presentation` (red takeaway callouts, metric cards, appendix-after-thanks, presenter notes, Docker/nginx on port 9000).

## Locked decisions (from 10 Q&A answers)

| Topic | Decision |
|---|---|
| Pacing | 45 min talk + 15 Q&A; **~38 content slides** (user presents fast; asked 30–45) + ~6 appendix |
| Opening | Outcome-dashboard cold-open ("3 years in numbers") right after title |
| Stack | **React 18 + Vite + Tailwind v4** (`@tailwindcss/vite`), no chart libs — hand-rolled SVG visuals |
| Interactives | Real-vs-sim Gantt · animated ROI waterfall · μ±σ explorer · architecture step-through · **model-improvement-over-time** (user-added) |
| Figures | Hybrid: rebuild web-native except the 2 densest diagrams (`03-customer-twin-overview.svg`, `05-transformer-architecture.svg`) embedded as-is |
| Branding | Handover palette canon: Navy `#003087`, Teal `#009999`, Orange `#EC6602` (callout accent), Light `#F0F7F7`, Mid `#CCEEEE`, Dark `#1A1A1A`, Muted `#6B7B7B`; "Siemens Healthineers" as text only |
| Ownership | Spotlight: Alternating Pipeline models, Databricks pipeline, MRRT agent, thesis MAS ("with Sebastian" = TODO confirm credit); VR = team |
| 3-yr story | Thematic phases with TODO year chips |
| Thesis scope | Dedicated 4-slide spotlight section |
| Demo assets | Web-native recreations captioned "illustrative" + labeled screenshot drop-in slots; best-effort real screenshot of agentic frontend |

## Fact inventory (only these numbers may appear on slides)

**Problem:** scans 30 s–45 min on one scanner; overruns cascade; $25K/machine/yr downtime+inefficiency (industry analysis); literature (Silva-Aravena et al. 2025): −44.8% patient wait, +14.5% utilization. Duration ranges table (HEAD scout 0.5–2 … ABDOMEN 12–30 min).
**System:** Four Twins (Device/Patient/Factory/**Customer=active**). 3 workstreams: Alternating Pipeline, MRRT Insight Agent, VR (Unity/Meta Quest). Data: 40 scanners, Apr–May 2024 event logs, 18-token vocab, 11 body regions, 15 sequence types. White-box 3-tier transformer (Conditioning Encoder: patient ×5/temporal ×5/coil ×30 + conditioning scale buffer → autoregressive Token Decoder: vocab 18, causal, cross-attn → bidirectional Duration Head: log-space μ±σ); d_model=128, 3 layers, 4 heads, FF=512; loss = token_CE + 0.3×duration_MSE. Alternation: Exchange ↔ Examination cycled by Orchestration (BREAK tokens). Validated decisions: log-space durations; conditioning scale buffer (LayerNorm erasure); separate Exchange/Exam; orchestrator needed (no GT patients at inference); bidirectional duration head.
**Results:** region order match >78% of days; day length ±12% (median, 40 scanners); exchange median error <15 s; duration MAE <1.8 min / median <1.1 min / σ within 20% / token acc >82%; per-region obs/pred μ (min): HEAD 9.0/9.2, NECK 7.0/7.4, SPINE 22.0/21.8, PELVIS 16.0/15.8, ABD 18.0/18.5; orchestrator edit distance 1.4 tokens/day, BREAK accuracy 89%; uncertainty ex.: HEAD μ8.0 σ1.2 vs SPINE μ18.0 σ5.0. MRRT: coil-detection = #1 SPINE friction; spine-coil override 3× head; pain table (coil 312 / interrupts 204 / table 178 / comfort 145 / image 97 / abort 89 — "example output" caveat).
**Ops/cloud:** local `run_all.py` steps 1→7 ↔ Databricks `spark_pipeline.py` (applyInPandas, 40 scanners parallel, Parquet + object storage, live, instrumented) — **same model weights both envs**. Status: 1,1b,2,3,2c DONE; 4,4b REFINING; 5 PLANNED.
**Scale/roadmap:** multi-site = shared backbone + adapter heads (≤4 wks local data), customer-ID embedding, federated-capable, cold-start <2 h CPU / ~500 days history. Next steps ×6 incl. orchestrator gate (edit distance ≤2.0), bucket pre-generation (1,000/bucket → instant assembly), Databricks hardening, fine-tuning registry, MRRT SPINE-coil report, external pilot. Roadmap: Q3'26 Model Stabilisation (NOW) → Q4'26 Customer Intelligence → Q1'27 Integration & Validation → Q2'27 Scale & Deploy (SaaS, portal).
**Financial:** waterfall $25K → −$7K downtime → −$6.2K throughput → −$4.5K wait → $7.3K residual ⇒ **~$17,700/machine/yr (71%)** with estimates-disclaimer; cases: coil-ROI pre-purchase, predictive maintenance (~$15K/event, 2-wk warning), VR ($500/h, 80% hours off-scanner). Ask: pilot site, ROI protocol, tech walkthrough, data agreement.
**Thesis (agentic-infra-copilot):** tripartite MAS (Governance/Hardware/Telemetry agents, per-domain ChromaDB RAG, autonomy protocol ANSWER/PARTIAL/CONSULT/…, synthesizer), DSPy + FastAPI microservices; LLM tiers: Haiku 4.5 reasoner / Sonnet 4.6 synthesizer / GPT-4.1 router. Siemens hook: ~50 real SHS PDFs (MAGNETOM Vida/Sola manuals, DICOM conformance, safety SOPs; ~40 customer installations context); stores: Hardware 9,853 / Telemetry 945 / Governance 418 docs. **Final results (thesis_summary.json, 2026-04-09, 12 cases × 5 modes):** Full MAS 88.3% keyword acc / 0.942 LLM-judge vs strongest single-agent+all-data baseline 83.3% / 0.855 ⇒ +5.0 pp; emergence 4/12 cases; single-agent ceilings Hardware 71.7% / Telemetry 55% / Governance 50%; latency MAS ~127 s vs ~59 s. React/TS demo UI (delegation badges, reasoning trace, results pages).
**Improvement-over-time (user-requested visual; numbers from repo validation history — verify before presenting):** Stage 1 exam-collapse fix: 26 → 21,620 generated exam rows; Stage 2 region realism: phantom-region collapse eliminated, region entropy 0.808, plausible mix; Stage 3 duration calibration: flat ~50 s for all types → per-type calibrated (synth mean 1.7 min ≈ real 1.75 min; scout 17 s vs real 19 s). Mark slide with a visible "TODO verify vs validation reports" chip.

## Narrative & slide-by-slide plan (~45 min, 38 slides + appendix)

Per slide: **Title — message | content | visual | red one-liner (orange accent)**. Every act's last slide carries a "So what" bar. Section kicker labels replace separator slides (fast pacing).

### Act 0 — Cold open (3 min, S1–S3)
- **S1 Title** — "MRI Digital Twin — 3 Years of Generative Simulation & Semantic Intelligence". Navy hero, teal edge bar, circles motif (port deck's title design). `TODO: presenter name/title/date line`.
- **S2 Three Years in Five Numbers** — the whole talk in one dashboard | 5 MetricCards: 3 generative models live on 40 scanners · ±12% day-length fidelity · <1.8 min duration MAE · ~$17.7K/machine/yr recoverable · 88.3% multi-agent diagnostic accuracy (thesis) | red: "Every number on this slide is measured, not promised — the next 45 minutes is the story behind each."
- **S3 Agenda** — 7 acts with per-act time chips | compact numbered grid (navy/orange split like deck).

### Act 1 — The mandate & the problem (5 min, S4–S8)
- **S4 The Four Twins** — we own the Customer Twin mandate | native rebuild: 4 cards, Customer Twin highlighted "ACTIVE PROJECT" | red: "Strategic mandate, not a side project."
- **S5 The Scheduling Problem** — stochastic process, deterministic slots | duration-range table + cascade illustration | red (deck quote): "Static scheduling treats a stochastic process as deterministic — the wrong model for MRI workflows."
- **S6 The Hidden Cost** — $25K/machine/yr | native rebuild of problem-stats: 4 stat tiles ($25K, −44.8%, +14.5%, 2 wks) with sources | red: "The literature already proves digital twins recover this — someone will build it."
- **S7 From Reporting to Simulation** — the strategic pivot | Today vs Tomorrow two-column block compare | red: "This transition is the Four Twins mandate."
- **S8 Gap → What We Build** — six gaps, six answers | condensed 6-row mapping table (current → needed → built) | So-what bar: "The gap is a generative, uncertainty-aware twin. We built it."

### Act 2 — What we built in 3 years (6 min, S9–S13)
- **S9 The 3-Year Journey** — four thematic phases | PhaseCards: Foundation (data) → Architecture (white-box transformer) → Validated Models → Cloud & Scale, each with 2 bullets + `TODO year` chips | red: "Each phase de-risked the next."
- **S10 Customer Twin System Overview** — inputs → 3 models → 5 outputs | **embed `03-customer-twin-overview.svg`** | red: "White-box AI: every prediction carries uncertainty bounds."
- **S11 Three Parallel Workstreams** — one data source, three products | 3 cards (Pipeline / MRRT / VR) | red: "All three draw on the same event-log foundation."
- **S12 Data Foundation** — structure from 40 scanners | native stat tiles (40 scanners · 2 months · 18 tokens · 11 regions · 15 types) + serials strip | red: "Thousands of timestamped hardware events per scanner-day."
- **S13 Our Contributions** — ownership map | spotlight cards: Alternating Pipeline models / Databricks pipeline / MRRT agent / thesis MAS (`TODO: Sebastian credit wording`), VR = team-attributed; `TODO: team size` chip | So-what: "The generative core and its cloud path are my direct work."

### Act 3 — How it works (7 min, S14–S18)
- **S14 The Alternating Pipeline** — a day is generated by alternation | native animated rebuild: Exchange↔Exam token chain with handoff tokens (T:100/T:104), conditioning inputs strip | red: "One model can't do this — alternation is the architecture."
- **S15 Inside the Model (interactive step-through)** — 3 tiers, plain language | **ArchitectureStepThrough**: Conditioning Encoder → Token Decoder → Duration Head, one sentence per tier, fragment-stepped; specs chip (d_model 128 · 3 layers · 4 heads) | red: "Full detail diagram in appendix — this is the mental model."
- **S16 White-Box vs Black-Box** — trust is a feature | two-column compare | red (deck): "Clinicians and engineers can understand every output."
- **S17 Uncertainty That Means Something (interactive)** — σ → schedule slack | **UncertaintyExplorer**: HEAD (μ8, σ1.2) vs SPINE (μ18, σ5) distributions + slack slider with buffer-minutes readout | red: "High σ isn't model weakness — it's honest scheduling guidance."
- **S18 Hard-Won Architecture Decisions** — validated by failure | 5-row decisions table (decision/why/evidence) | So-what: "These scars are why the results in the next act hold."

### Act 4 — Results (9 min, S19–S24) ← the heart
- **S19 A Simulated Day vs Reality (interactive)** — the signature proof | **GanttComparison**: ground-truth vs simulated day lanes (HEAD/SPINE/PELVIS/ABD + exchange slivers, 08:00–12:00), play button assembles sim | red: "Same scanner, same date — structure reproduced."
- **S20 Day-Level Fidelity** — three numbers that matter | MetricCards: >78% region-order match · ±12% day length (median, 40 scanners) · <15 s exchange median error | red: "Good enough to schedule against."
- **S21 Duration Prediction** — mean AND spread | **RegionBarChart** (obs vs pred, 5 regions) + metrics table (MAE <1.8 / median <1.1 / σ within 20% / token acc >82%) | red (deck): "Predicting the spread enables principled slack allocation."
- **S22 Orchestrator Validation** — autonomous day sequencing | **SequenceCompare** strip (GT vs predicted tokens, 1 mismatch, edit distance 1) + stats (mean 1.4 tokens/day, BREAK 89%) | red: "No ground-truth patient list needed at inference."
- **S23 How the Model Improved (interactive)** — from collapse to calibration | **ImprovementTimeline**: Stage 1 26→21,620 rows · Stage 2 phantom regions eliminated (entropy 0.808) · Stage 3 flat→calibrated durations (1.7 vs 1.75 min real); visible `TODO: verify vs validation reports` chip | red: "Every stage had a measurable gate — and passed it."
- **S24 MRRT Insight Agent Results** — semantic layer already pays | before→after transform mini-diagram (unstructured comment → structured pain record) + pain-category table with "example output" caveat + 3× override stat | So-what: "R&D-actionable signals from text nobody could read at scale."

### Act 5 — Thesis spotlight: Agentic Infra Co-Pilot (5 min, S25–S28)
- **S25 What It Is** — second act of AI on Siemens data | mission card + data tiles (~50 SHS PDFs: MAGNETOM manuals, DICOM conformance, safety SOPs; 9,853-doc hardware store) + master-thesis framing (`TODO: Sebastian credit`, `TODO: confirm numbers match thesis Ch. 4`) | red: "Same Siemens fleet — now diagnosing faults, not simulating schedules."
- **S26 Three Specialists, One Diagnosis** — MAS architecture | native diagram: Governance/Hardware/Telemetry agents (per-domain RAG) → autonomy protocol → synthesizer; LLM-tier chips | red: "Each agent can say 'not my domain' — that's the protocol, not a failure."
- **S27 Measured Emergence** — collaboration beats the best single agent | grouped bar: single-agent ceilings (71.7/55/50) vs single+all-data 83.3% vs MAS 88.3%; judge 0.942 vs 0.855; emergence 4/12; latency honesty chip (127 s vs 59 s) | red: "+5.0 pp over the strongest baseline — same data, same models, the delta is collaboration."
- **S28 Demo & Why It Matters Here** — from simulation to operations support | ScreenshotSlot (agentic chat UI w/ delegation badges — best-effort real capture) + "what we can show live" list | So-what: "The twin predicts the day; the copilot explains the fleet. Same data estate."

### Act 6 — Operations & cloud potential (4 min, S29–S31)
- **S29 How It Runs Today** — dual-environment parity | two-column Local CPU ↔ Databricks + pipeline-status strip (DONE/REFINING/PLANNED chips) | red (deck): "Both environments produce identical model artefacts."
- **S30 Cloud Readiness** — scale-up, not rewrite | readiness indicator board: parity ✓ · 40-scanner parallel preprocessing ✓ · Parquet/object storage ✓ · step instrumentation ✓ · hardening ◐ · bucket pre-generation ○ (→ instant day assembly) + expected benefits column | red: "The same weights already run in both worlds."
- **S31 Scaling Architecture** — single customer → fleet | backbone + adapter-heads diagram; cold-start facts (<2 h CPU, ~500 days, ≤4 wks local data, federated-capable) | So-what: "Conditioning was designed for this — customer ID slots in without architectural change."

### Act 7 — Roadmap, money, ask (6 min, S32–S38)
- **S32 Development Roadmap** — Q3'26 → Q2'27 | native 4-column RoadmapColumns with NOW marker + revision disclaimer | red: "Next two quarters are committed; the rest is sequenced."
- **S33 Immediate Next Steps** — six blocks with gates | 6 cards incl. explicit gate chips (edit distance ≤2.0; 1,000/bucket) | red: "Each step has a measurable exit criterion."
- **S34 Financial Impact (interactive)** — the waterfall | **RoiWaterfall** animated: $25K → … → $7.3K residual; banner ~$17.7K/yr (71%); sources+estimates disclaimer kept visible | red: "71% of a known, quantified loss is recoverable."
- **S35 Three Business Cases** — concrete monetisation | 3 cards (coil-ROI pre-purchase / predictive maintenance ~$15K events, 2-wk warning / VR $500/h, 80% off-scanner) | red: "Each case reuses the same twin — zero extra model builds."
- **S36 Three Things to Remember** — the Duarte landing | numbered takeaway list (working twin · built to scale · quantified ROI) | red: "If you remember one: it works, measured, on 40 real scanners."
- **S37 The Ask** — decision slide | 4 action cards: pilot site · ROI measurement protocol · technical walkthrough · data-sharing agreement | So-what: "Decision requested today: pilot site + data agreement."
- **S38 Thank You / Q&A** — contact + appendix pointer | navy hero mirror of S1 | red: "Appendix: full architecture, run-book, methods — press → to dive."

### Appendix (Q&A on-demand, A1–A6)
- **A1** Full transformer architecture — **embed `05-transformer-architecture.svg`**
- **A2** `run_all.py` runbook — steps 1→7 table
- **A3** Architecture-decisions detail + duration-range table
- **A4** MRRT method detail (embedding + retrieval + LLM; structured schema)
- **A5** Thesis eval design (12 cases × 5 modes; RQ1–RQ3; results-provenance note: final JSON vs stale README)
- **A6** Sources & disclaimers (Silva-Aravena et al. 2025, industry analysis) + Missing-Info/TODO index

## Implementation

**Location:** `presentation/web/` in this repo (new folder; LaTeX deck untouched).

```
presentation/web/
├─ package.json            # react, react-dom, vite, @vitejs/plugin-react, tailwindcss@4, @tailwindcss/vite, @fontsource-variable/inter (offline), vitest
├─ vite.config.ts · index.html · README.md (run guide + screenshot drop-ins + TODO index)
├─ Dockerfile              # multi-stage: node:20-alpine build → nginx:alpine; docker-compose.yml → port 9000 (reference parity)
├─ public/assets/          # copied 03-customer-twin-overview.svg, 05-transformer-architecture.svg; screenshots/ drop-in dir
└─ src/
   ├─ main.tsx · App.tsx · theme/tokens.css   # SHS palette as CSS vars + Tailwind theme
   ├─ deck/                # engine: DeckProvider (slide+fragment state, #/n hash sync, ←→/Space/Esc/N keys), SlideChrome (kicker, act label, progress bar, n/total), Overview grid (Esc), SpeakerNotes panel (N: notes + act time budget + elapsed timer), print.css (PDF fallback)
   ├─ components/          # MetricCard, StatTile, RedCallout (orange one-liner w/ label), SoWhatBar, SHSTable, PhaseCard, RoadmapColumn, StatusChip, TodoChip, ScreenshotSlot, SectionKicker, CompareColumns
   ├─ viz/                 # hand-rolled SVG + CSS transitions (no chart lib): GanttComparison, RoiWaterfall, UncertaintyExplorer, ArchitectureStepThrough, ImprovementTimeline, RegionBarChart, SequenceCompare
   ├─ slides/              # one file per slide s01…s38, a01…a06 + index.ts registry (order, act metadata, speaker notes, fragment counts)
   └─ data/facts.ts        # ALL slide numbers/strings from the Fact Inventory in one typed module (single point of truth, easy fact-audit)
```

**Engine behavior:** arrow keys advance fragments then slides; hash deep-links `#/14`; Esc = overview grid (act-grouped thumbnails); N = notes; P or `?print` = print stylesheet. Fixed 16:9 stage scaled to viewport (reference-deck approach, avoids per-slide responsive bugs) with `overflow-x` safety.

**Order of work** (single session, staged): scaffold + engine + theme → shared components → slides by act (0→7) → viz interactives → appendix → README + Docker → verify. Load `frontend-design` skill before theme/component work and `dataviz` skill before viz components (per skill rules). After approval, save a project memory entry for this presentation effort. Best-effort: run agentic-infra-copilot frontend (`npm run dev`, static pages only) to capture a real Results/Architecture screenshot into `public/assets/screenshots/`.

**Not doing:** no invented metrics (facts.ts audited against Fact Inventory); no logo files; no external CDNs (fonts vendored via npm); VR content stays team-attributed; LaTeX deck untouched.

## Verification

1. `npm run build` passes clean (TS strict) + `npx vitest run` (engine: registry integrity — every slide has act/notes/fragment count; hash↔slide mapping; fragment ordering).
2. `npm run dev` → walk all 44 slides by keyboard; check: cold-open dashboard, all 7 interactives respond (Gantt play, waterfall animation, σ slider, step-through fragments, improvement timeline), Esc overview, N notes+timer, `#/27` deep link, print stylesheet.
3. `docker compose up` → deck serves on `localhost:9000` (parity with reference repo).
4. Fact audit pass: grep every number in `data/facts.ts` against the Fact Inventory section above.
5. Screenshot set of ~6 key slides sent to Luke for visual sign-off.

## Missing-info list (rendered as TODO chips on slides + A6 index + README)

1. Year markers for the four journey phases (S9).
2. Presenter name/title/date line (S1); audience/team names optional.
3. **Sebastian's credit wording** — he appears nowhere in the agentic repo, git history (all 81 commits are Luke's), or thesis acknowledgements (supervisors: Iolanda Velho, Nuno Correia). Confirm role before he's named on S13/S25.
4. Improvement-over-time stage numbers (S23) — verify against pipeline validation reports before presenting.
5. Thesis results (S27) — confirm `thesis_summary.json` figures match final thesis Chapter 4 (README/RESULTS_SUMMARY hold stale weaker runs).
6. Real screenshots when available: Qlik dashboard, Gantt HTML output, VR still, agentic chat UI (S28 slot; best-effort capture during build).
7. Team size / FTE context for S13 (optional credibility anchor).
8. Confirm "3 years" phrasing for the intro (deck is dated June 2026; training data Apr–May 2024).

---

## Sign-off round (July 2026) — resolutions applied

- **Presenter**: Luke Schumacher · SHS DI D&A · July 2026 (S1/S38).
- **Journey years** (S9): 2024 Foundation · 2025 Architecture · 2026 Validated models · 2026 Cloud & scale; callout notes that many model variants were tried, evaluated and retired (version history = audit trail).
- **Team** (S13): Team of 4 — Luke Schumacher · Navneet · Martina · Georg (team lead). Sebastian credited for MRRT corpus + feedback (S13 card, S25 lede).
- **S23 verification**: before-states + gates confirmed in repo code (validate_stage1_examination.py 26-row baseline; examination_duration_calibration.py flat-prior mechanism; step-05 entropy gate). After-numbers from June 2026 validation runs; artifacts on Databricks DBFS → provenance chip replaces the TODO.
- **S27 verification**: confirmed against results/ablation/thesis_summary.json AND thesis **Chapter 5**; cross-domain chip added (Telekom IoT: MAS +6 pp judge, +33 pp semantic — thesis-cited); full Telekom Run-4 Pareto table on A5 with the merge-strategy lesson.
- **S34**: bracket clipping fixed (wider canvas); fleet-size slider added at the final fragment (n × $17.7K/yr; 40 machines ≈ $708K/yr).
- **New visuals**: S05 duration-range chart (70× spread annotation), S31 backbone→adapter diagram.
- **Engine**: `#/34.4`-style fragment deep links.
