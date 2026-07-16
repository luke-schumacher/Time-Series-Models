import { SHSTable } from '../components/blocks';
import { RedCallout, SlideTitle, TodoChip } from '../components/core';
import { IMPROVEMENT_PROVENANCE, PROBLEM, ROI } from '../data/facts';

const OPEN_ITEMS: readonly (readonly [string, string])[] = [
  ['Team names (S13)', 'confirm spelling — Navneet (from git history) · Martina · Georg'],
  ['Demo slides (optional)', 'extra screenshots: Qlik dashboard · Gantt HTML · VR still · mid-diagnosis agentic shot'],
];

export function A06Sources() {
  return (
    <>
      <SlideTitle>Appendix · sources, disclaimers &amp; open items</SlideTitle>
      <div className="grid flex-1 grid-cols-2 items-start gap-8">
        <div>
          <div className="font-mono text-[11.5px] tracking-[0.13em] text-navy uppercase">Sources</div>
          <ul className="mt-3 space-y-2.5 text-[15px] leading-snug text-ink/85">
            <li>
              <strong>{PROBLEM.literatureSource}</strong> — digital-twin waitlist management
              (−44.8% wait) and RL utilisation gains (+14.5%).
            </li>
            <li>
              <strong>{PROBLEM.baselineLossSource}</strong> — $25K/machine/year downtime and
              inefficiency baseline.
            </li>
            <li>
              <strong>Handover deck (June 2026)</strong> — all model metrics, architecture and
              roadmap content.
            </li>
            <li>
              <strong>thesis_summary.json (2026-04-09)</strong> — all Agentic Infra Co-Pilot
              figures; confirmed against thesis Chapter 5.
            </li>
            <li>
              <strong>IoT-IPE Copilot deck (T-Labs, Run 4, 2026-04-07)</strong> — the Telekom
              cross-domain ablation on appendix A5.
            </li>
            <li>
              <strong>Stage-gate numbers (S23)</strong> — {IMPROVEMENT_PROVENANCE}.
            </li>
          </ul>
          <p className="mt-4 rounded-md bg-surface px-4 py-3 font-mono text-[12px] leading-relaxed text-muted">
            {ROI.disclaimer}
          </p>
        </div>
        <div>
          <div className="font-mono text-[11.5px] tracking-[0.13em] text-orange-ink uppercase">
            Open items before presenting
          </div>
          <div className="mt-3">
            <SHSTable
              head={['Where', 'What']}
              rows={OPEN_ITEMS.map(([w, what]) => [w, <TodoChip key={w}>{what}</TodoChip>])}
              className="text-[13px]"
            />
          </div>
        </div>
      </div>
      <RedCallout label="Integrity">
        Every number in this deck traces to one of the four sources on the left — nothing was
        invented for the slides.
      </RedCallout>
    </>
  );
}
