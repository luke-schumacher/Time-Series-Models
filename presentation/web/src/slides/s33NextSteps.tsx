import { Pill, RedCallout, SlideTitle } from '../components/core';
import { NEXT_STEPS } from '../data/facts';

export function S33NextSteps() {
  return (
    <>
      <SlideTitle>Immediate next steps — with exit gates</SlideTitle>
      <div className="grid flex-1 grid-cols-3 gap-5">
        {NEXT_STEPS.map((s) => (
          <div key={s.n} className="flex flex-col rounded-lg border border-mist bg-paper p-5 shadow-sm">
            <div className="flex items-center justify-between gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-md bg-navy font-mono text-[13px] font-bold text-white">
                {s.n}
              </span>
              <Pill tone="teal">{s.gate}</Pill>
            </div>
            <h3 className="font-display mt-3 text-[18px] leading-tight font-bold text-navy">{s.title}</h3>
            <p className="mt-2 text-[13.5px] leading-snug text-ink/80">{s.desc}</p>
          </div>
        ))}
      </div>
      <RedCallout>
        Every step has a measurable exit criterion — nothing on this slide can be "in progress
        forever."
      </RedCallout>
    </>
  );
}
