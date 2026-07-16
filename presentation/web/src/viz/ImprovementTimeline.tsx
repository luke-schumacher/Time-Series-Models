import { useFragment } from '../deck/DeckContext';
import { IMPROVEMENT } from '../data/facts';

/** Stage 1 gets a log-scale before/after bar — the jump is 3 orders of magnitude. */
function LogBars({ before, after }: { before: number; after: number }) {
  const w = (v: number) => (Math.log10(v) / Math.log10(after)) * 100;
  return (
    <div className="mt-2 space-y-1">
      <div className="flex items-center gap-2">
        <div className="h-3 rounded-r bg-mist" style={{ width: `${w(before)}%` }} />
        <span className="font-mono text-[11px] text-muted">{before.toLocaleString('en-US')}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="h-3 rounded-r bg-teal" style={{ width: `${w(after)}%` }} />
        <span className="font-mono text-[11px] font-semibold text-teal-ink">
          {after.toLocaleString('en-US')}
        </span>
      </div>
      <div className="font-mono text-[10px] text-muted">log scale</div>
    </div>
  );
}

/** S23: three validated stages from collapse to calibration (fragments = 2). */
export function ImprovementTimeline() {
  const visible = [true, useFragment(1), useFragment(2)];

  return (
    <div className="relative flex flex-1 items-stretch gap-6">
      {/* spine */}
      <div className="absolute top-[26px] right-4 left-4 h-[3px] rounded bg-mist" aria-hidden />
      {IMPROVEMENT.map((s, i) => (
        <div key={s.stage} className="frag relative flex flex-1 flex-col" data-visible={visible[i]}>
          <div className="z-10 flex items-center gap-3">
            <span className="grid h-[52px] w-[52px] place-items-center rounded-full bg-navy font-mono text-[15px] font-bold text-white ring-4 ring-paper">
              {i + 1}
            </span>
            <span className="font-mono text-[12px] font-semibold tracking-[0.12em] text-teal-ink uppercase">
              {s.stage}
            </span>
          </div>
          <div className="mt-4 flex flex-1 flex-col rounded-lg border border-mist bg-paper p-5 shadow-sm">
            <h3 className="font-display text-[20px] leading-tight font-bold text-navy">{s.title}</h3>
            <div className="mt-3 space-y-2">
              <div>
                <div className="font-mono text-[10.5px] tracking-[0.12em] text-muted uppercase">before</div>
                <div className="text-[15.5px] text-muted line-through decoration-orange/60 decoration-2">
                  {s.before}
                </div>
              </div>
              <svg width="16" height="14" viewBox="0 0 16 14" className="text-teal" aria-hidden>
                <path d="M8 0v10m0 0l-5-4.5M8 10l5-4.5" stroke="currentColor" strokeWidth="2" fill="none" />
              </svg>
              <div>
                <div className="font-mono text-[10.5px] tracking-[0.12em] text-teal-ink uppercase">after</div>
                <div className="kpi-number text-[24px] leading-tight font-bold text-teal-ink">{s.after}</div>
              </div>
            </div>
            {'beforeNum' in s && 'afterNum' in s && <LogBars before={s.beforeNum} after={s.afterNum} />}
            <p className="mt-3 border-t border-mist pt-2.5 text-[13px] leading-snug text-ink/75">{s.detail}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
