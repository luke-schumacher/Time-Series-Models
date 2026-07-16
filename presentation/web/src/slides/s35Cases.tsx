import { RedCallout, SlideTitle } from '../components/core';
import { BUSINESS_CASES } from '../data/facts';

const FIELDS = [
  { key: 'use', label: 'Use case' },
  { key: 'how', label: 'Solution' },
  { key: 'result', label: 'Result' },
] as const;

export function S35Cases() {
  return (
    <>
      <SlideTitle>Three concrete business cases</SlideTitle>
      <div className="grid flex-1 grid-cols-3 gap-6">
        {BUSINESS_CASES.map((c) => (
          <div key={c.n} className="flex flex-col rounded-lg border-t-4 border-mist border-t-navy bg-paper p-5 shadow-sm">
            <h3 className="font-display text-[19.5px] leading-tight font-bold text-navy">
              {c.n}. {c.title}
            </h3>
            <div className="mt-3.5 flex flex-1 flex-col gap-3">
              {FIELDS.map((f) => (
                <div key={f.key}>
                  <div className="font-mono text-[10.5px] tracking-[0.13em] text-teal-ink uppercase">
                    {f.label}
                  </div>
                  <p className={`mt-0.5 text-[14px] leading-snug ${f.key === 'result' ? 'font-semibold text-ink' : 'text-ink/80'}`}>
                    {c[f.key]}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <RedCallout>
        Each case reuses the same trained twin with different conditioning — zero additional model
        builds.
      </RedCallout>
    </>
  );
}
