# MRI Digital Twin — 3-Year Results Presentation

Interactive web deck for the boss's-boss results report: three years of the MRI Digital Twin
(Customer Twin) programme plus the Agentic Infra Co-Pilot master thesis. 38 core slides + 6
appendix slides, ~45 min talk + 15 min Q&A.

Built with React 18 + Vite + Tailwind v4. Fully offline — fonts and figures are vendored; no
CDN calls.

## Run it

```bash
# development
npm install
npm run dev            # http://localhost:5173

# production build + preview
npm run build
npm run preview        # http://localhost:4173

# docker (parity with the reference deck: port 9000)
docker compose up --build   # http://localhost:9000
```

## Controls

| Key | Action |
|-----|--------|
| `→` `Space` `PgDn` | next fragment / slide |
| `←` `PgUp` | previous fragment / slide |
| `Esc` / `O` | section overview grid (click to jump) |
| `N` | presenter notes + act time budget + elapsed clock |
| `F` | fullscreen |
| `Home` / `End` | first / last slide |

- Deep links: `#/14` (slide number) or `#/s14` / `#/a02` (slide id); `#/34.4` opens a slide at
  fragment 4 (useful for jumping to a fully built chart).
- PDF fallback: open `/?print`, then print (A4 landscape) — all fragments fully revealed.
- The bar at the top of every slide is the talk itself rendered as a scanner-day schedule:
  act blocks sized by time budget, teal scan line = current position.

## Structure

```
src/
├─ data/facts.ts     ← SINGLE SOURCE OF TRUTH for every number on a slide
├─ deck/             ← engine: navigation, hash routing, overview, notes, print
├─ components/       ← metric cards, red callouts, so-what bars, tables, chips
├─ viz/              ← hand-rolled SVG interactives (Gantt, waterfall, μ±σ, …)
└─ slides/           ← one file per slide (s01…s38, a01…a06) + registry (index.ts)
```

Fact provenance (see also appendix slide A6):

- Handover deck: `presentation/main.tex` + `presentation/svgs/` — all Digital-Twin metrics.
- Thesis: `agentic-infra-copilot/results/ablation/thesis_summary.json` (2026-04-09) — the
  README/RESULTS_SUMMARY tables in that repo are **stale earlier runs**; do not quote them.
- Model-improvement slide (S23): repo validation history — **verify before presenting**.

## Status of the former TODOs (resolved July 2026, indexed on slide A6)

1. **S1/S38** — ✅ Luke Schumacher · SHS DI D&A · July 2026.
2. **S9** — ✅ years set: 2024 (Foundation), 2025 (Architecture), 2026 (Validation, Cloud & scale).
3. **S13/S25** — ✅ team of 4 (Luke Schumacher · Navneet · Martina · Georg, team lead); Sebastian
   credited for MRRT data access & feedback. *Check the spelling "Navneet" (taken from git
   history; the sign-off note wrote "Naveet").*
4. **S23** — ✅ verified as far as the repo allows: before-states and gates are in code
   (`validate_stage1_examination.py`, `examination_duration_calibration.py`, step-05 entropy
   gate); the after-numbers are from the June 2026 validation runs whose artifacts live on
   Databricks DBFS. The slide now carries a provenance chip instead of a TODO.
5. **S27** — ✅ confirmed against `thesis_summary.json` AND the thesis (results are
   **Chapter 5**); Telekom cross-domain line added (thesis-cited), full Pareto detail on A5.
6. **Screenshots** — `agentic-chat-ui.png` live capture installed (idle state). Optional
   additions for S28/appendix: mid-diagnosis agentic shot, Qlik dashboard, Gantt HTML, VR still.

## Editing facts

Never type a number directly in a slide. Add or change it in `src/data/facts.ts`, where every
constant is annotated with its source. `npm test` checks registry integrity (unique ids, act
coverage, notes present, fragment sanity).
