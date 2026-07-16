import { SHSTable } from '../components/blocks';
import { SoWhatBar, SlideTitle } from '../components/core';
import { MRRT } from '../data/facts';

export function S24Mrrt() {
  return (
    <>
      <SlideTitle lede="LLM semantic retrieval over thousands of free-text customer comments.">
        MRRT Insight Agent — what it found
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.05fr_1fr] gap-8">
        {/* transform demo */}
        <div className="flex flex-col justify-center gap-2.5">
          <div className="rounded-lg border border-mist bg-surface/70 px-5 py-3.5 text-[15px] text-ink/75 italic">
            {MRRT.exampleIn}
          </div>
          <div className="flex items-center gap-3 pl-4">
            <svg width="14" height="26" viewBox="0 0 14 26" className="text-teal" aria-hidden>
              <path d="M7 0v20m0 0l-6-6m6 6l6-6" stroke="currentColor" strokeWidth="2.2" fill="none" />
            </svg>
            <span className="rounded-md bg-navy px-3.5 py-1.5 font-mono text-[12px] font-semibold text-white">
              MRRT Insight Agent <span className="ml-1 text-teal">{MRRT.engine}</span>
            </span>
          </div>
          <div className="rounded-lg border border-teal/50 bg-teal/[0.05] px-5 py-3.5 font-mono text-[13.5px] leading-relaxed">
            {MRRT.exampleOut.map(([k, v]) => (
              <div key={k}>
                <span className="text-muted">{k}:</span>{' '}
                <span className="font-semibold text-teal-ink">{v}</span>
              </div>
            ))}
          </div>
          <p className="mt-2 text-[15px] leading-snug text-ink/85">
            <strong className="text-navy">Key finding:</strong> {MRRT.findings[0]} — override rate{' '}
            <strong className="text-orange-ink">3× higher</strong> for spine coils than head coils,
            a signal structured surveys missed entirely.
          </p>
        </div>

        {/* pain table */}
        <div className="flex flex-col justify-center">
          <SHSTable
            head={['Pain category', 'Count', 'Impact']}
            rows={MRRT.pains.map((p) => [
              p.cat,
              p.count,
              <span className={p.impact === 'High' ? 'font-semibold text-orange-ink' : 'text-muted'}>
                {p.impact}
              </span>,
            ])}
            mono={[1]}
          />
          <p className="mt-2 text-right font-mono text-[11.5px] text-muted">{MRRT.caveat}</p>
        </div>
      </div>
      <SoWhatBar>
        R&amp;D-actionable signals from text nobody could read at scale — already shaping coil
        firmware priorities.
      </SoWhatBar>
    </>
  );
}
