import { RedCallout, SlideTitle } from '../components/core';
import { SEBASTIAN_CREDIT, THESIS } from '../data/facts';

export function S25ThesisIntro() {
  return (
    <>
      <SlideTitle
        lede={
          <>
            <span className="mr-2 rounded-full bg-navy px-2.5 py-0.5 font-mono text-[11px] font-bold tracking-wider text-white">
              MASTER THESIS
            </span>
            {THESIS.framing} ·{' '}
            <span className="font-medium text-teal-ink">{SEBASTIAN_CREDIT}</span>
          </>
        }
      >
        Second act on the same data: {THESIS.name}
      </SlideTitle>

      <div className="grid flex-1 grid-cols-[1.2fr_1fr] gap-8">
        <div className="flex flex-col justify-center gap-5">
          <p className="text-[20px] leading-relaxed text-ink">{THESIS.pitch}</p>
          <div className="rounded-lg border-l-4 border-navy bg-surface px-5 py-4">
            <div className="font-mono text-[11.5px] tracking-[0.13em] text-navy uppercase">
              Research question
            </div>
            <p className="mt-1.5 text-[18px] leading-snug font-semibold text-navy">
              {THESIS.question}
            </p>
          </div>
        </div>
        <div className="flex flex-col justify-center">
          <div className="rounded-lg border border-mist bg-paper p-6 shadow-sm">
            <div className="font-mono text-[11.5px] tracking-[0.13em] text-teal-ink uppercase">
              Validated on Siemens Healthineers data
            </div>
            <ul className="mt-3.5 space-y-2.5">
              {THESIS.siemensData.map((d) => (
                <li key={d} className="flex gap-2.5 text-[15.5px] leading-snug text-ink/85">
                  <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-teal" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      <RedCallout>
        Same Siemens fleet, opposite direction — the twin simulates the schedule; the copilot
        diagnoses the infrastructure behind it.
      </RedCallout>
    </>
  );
}
