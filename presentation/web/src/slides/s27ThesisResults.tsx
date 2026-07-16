import { RedCallout, SlideTitle } from '../components/core';
import { THESIS } from '../data/facts';
import { EmergenceBars } from '../viz/EmergenceBars';

export function S27ThesisResults() {
  const r = THESIS.results;
  return (
    <>
      <SlideTitle
        lede={`${r.setup} · keyword accuracy against expert-defined findings · confirmed against thesis Chapter 5`}
      >
        Measured emergence
      </SlideTitle>

      <EmergenceBars />

      <div className="mt-1 grid grid-cols-4 gap-4">
        <div className="rounded-lg bg-surface px-5 py-3">
          <div className="font-mono text-[11px] tracking-[0.12em] text-muted uppercase">LLM-judge score</div>
          <div className="mt-1 text-[19px] font-semibold text-ink">
            <span className="text-muted">{r.judgeBaseline.toFixed(3)}</span>
            <span className="mx-2 text-teal">→</span>
            <span className="kpi-number font-bold text-teal-ink">{r.judgeMas.toFixed(3)}</span>
          </div>
        </div>
        <div className="rounded-lg bg-surface px-5 py-3">
          <div className="font-mono text-[11px] tracking-[0.12em] text-muted uppercase">Emergent cases</div>
          <div className="mt-1 text-[19px] font-semibold text-ink">
            <span className="kpi-number font-bold text-teal-ink">{r.emergence}</span>
            <span className="ml-2 text-[14px] text-muted">solved by no single agent</span>
          </div>
        </div>
        <div className="rounded-lg bg-surface px-5 py-3">
          <div className="font-mono text-[11px] tracking-[0.12em] text-muted uppercase">Honest cost</div>
          <div className="mt-1 text-[19px] font-semibold text-ink">
            <span className="kpi-number font-bold text-navy">{r.latencyMas}</span>
            <span className="ml-2 text-[14px] text-muted">vs {r.latencyBaseline} single-agent</span>
          </div>
        </div>
        <div className="rounded-lg bg-surface px-5 py-3">
          <div className="font-mono text-[11px] tracking-[0.12em] text-muted uppercase">
            Cross-domain check
          </div>
          <div className="mt-1 text-[19px] font-semibold text-ink">
            <span className="kpi-number font-bold text-teal-ink">{THESIS.crossDomain.headline}</span>
          </div>
          <div className="mt-0.5 text-[12px] leading-tight text-muted">{THESIS.crossDomain.sub}</div>
        </div>
      </div>

      <RedCallout>
        +{r.marginPp.toFixed(1)} pp over the strongest single-agent baseline — same data, same
        models. The delta is collaboration.
      </RedCallout>
    </>
  );
}
