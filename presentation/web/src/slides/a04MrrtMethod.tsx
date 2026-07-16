import { RedCallout, SlideTitle } from '../components/core';
import { MRRT } from '../data/facts';

export function A04MrrtMethod() {
  return (
    <>
      <SlideTitle lede={`Pipeline: ${MRRT.engine} over the full comment corpus.`}>
        Appendix · MRRT Insight Agent — method
      </SlideTitle>
      <div className="grid flex-1 grid-cols-2 gap-8">
        <div>
          <div className="font-mono text-[11.5px] tracking-[0.13em] text-navy uppercase">
            The challenge
          </div>
          <ul className="mt-3 space-y-2.5">
            {MRRT.challenge.map((c) => (
              <li key={c} className="flex gap-2.5 text-[15.5px] leading-snug text-ink/85">
                <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-teal" />
                {c}
              </li>
            ))}
          </ul>
        </div>
        <div>
          <div className="font-mono text-[11.5px] tracking-[0.13em] text-navy uppercase">
            Structured output schema
          </div>
          <div className="mt-3 rounded-lg border border-teal/50 bg-teal/[0.05] px-5 py-4 font-mono text-[14px] leading-relaxed">
            {MRRT.exampleOut.map(([k]) => (
              <div key={k}>
                <span className="text-muted">{k}:</span>{' '}
                <span className="text-teal-ink">&lt;{k === 'frequency' ? 'low | medium | high' : 'extracted'}&gt;</span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[15px] leading-snug text-ink/85">{MRRT.impact}</p>
        </div>
      </div>
      <RedCallout label="Appendix">
        Ranked by occurrence and clinical impact — the ranking is the product, not the retrieval.
      </RedCallout>
    </>
  );
}
