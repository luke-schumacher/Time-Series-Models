import { RedCallout, SlideTitle } from '../components/core';
import { ARCHITECTURE } from '../data/facts';

export function A03ModelRef() {
  return (
    <>
      <SlideTitle lede={`${ARCHITECTURE.specs} · ${ARCHITECTURE.loss}`}>
        Appendix · model configuration reference
      </SlideTitle>
      <div className="grid flex-1 grid-cols-3 gap-5">
        {ARCHITECTURE.tiers.map((t) => (
          <div key={t.n} className="rounded-lg border border-mist bg-paper p-5">
            <div className="font-mono text-[11.5px] font-bold tracking-[0.13em] text-teal-ink">
              TIER {t.n}
            </div>
            <h3 className="font-display mt-1 text-[20px] font-bold text-navy">{t.name}</h3>
            <ul className="mt-3 space-y-2">
              {t.details.map((d) => (
                <li key={d} className="flex gap-2 text-[13.5px] leading-snug text-ink/80">
                  <span className="mt-[8px] h-[3px] w-2.5 shrink-0 rounded bg-teal/60" />
                  {d}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <RedCallout label="Training">
        Early stopping on a per-scanner validation split; duration loss masked to real targets;
        conditioning scale registered as a model buffer.
      </RedCallout>
    </>
  );
}
