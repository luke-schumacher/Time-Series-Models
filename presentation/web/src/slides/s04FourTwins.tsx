import { RedCallout, SlideTitle } from '../components/core';
import { FOUR_TWINS } from '../data/facts';

export function S04FourTwins() {
  return (
    <>
      <SlideTitle lede="Siemens' digital-platform strategy rests on four simulation pillars.">
        The Four Twins ecosystem
      </SlideTitle>
      <div className="relative mt-2">
        <div className="mx-auto w-fit rounded-md bg-navy px-6 py-2 font-mono text-[13px] font-semibold tracking-[0.14em] text-white uppercase">
          Siemens Digital Platform
        </div>
        <div className="mx-auto h-6 w-px bg-mist" />
        <div className="grid grid-cols-4 gap-6">
          {FOUR_TWINS.map((t) => (
            <div
              key={t.name}
              className={`flex flex-col rounded-lg border p-6 ${
                t.active
                  ? 'border-teal bg-teal/[0.06] shadow-md ring-1 ring-teal/30'
                  : 'border-mist bg-surface/60'
              }`}
            >
              <h3
                className={`font-display text-[22px] leading-tight font-bold ${
                  t.active ? 'text-navy' : 'text-muted'
                }`}
              >
                {t.name}
              </h3>
              <p className={`mt-2 flex-1 text-[15px] leading-snug ${t.active ? 'text-ink/85' : 'text-muted'}`}>
                {t.desc}
              </p>
              {t.active ? (
                <span className="mt-4 w-fit rounded-full bg-teal px-3 py-1 font-mono text-[11px] font-bold tracking-wider text-white">
                  ▶ ACTIVE PROJECT
                </span>
              ) : (
                <span className="mt-4 w-fit font-mono text-[11px] tracking-wider text-muted uppercase">
                  parallel pillar
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
      <RedCallout>
        The Customer Twin is a strategic mandate, not a side project — this report covers the
        pillar we own.
      </RedCallout>
    </>
  );
}
