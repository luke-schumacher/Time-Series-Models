import { useFragment } from '../deck/DeckContext';
import { ARCHITECTURE } from '../data/facts';

/** S15: the three-tier mental model, revealed tier by tier (fragments = 3). */
export function ArchStepThrough() {
  const tiers = ARCHITECTURE.tiers;
  const visible = [true, useFragment(1), useFragment(2)];
  const showLoss = useFragment(3);

  return (
    <div className="flex flex-1 flex-col gap-5">
      <div className="grid flex-1 grid-cols-[1fr_44px_1fr_44px_1fr] items-stretch gap-0">
        {tiers.map((t, i) => (
          <div key={t.n} className="contents">
            {i > 0 && (
              <div className="frag flex flex-col items-center justify-center gap-1" data-visible={visible[i]}>
                <svg width="30" height="18" viewBox="0 0 30 18" aria-hidden>
                  <path d="M0 9h24m0 0l-7-7m7 7l-7 7" stroke="#009999" strokeWidth="2.2" fill="none" />
                </svg>
                <span className="font-mono text-[10.5px] tracking-wide text-muted">
                  {i === 1 ? 'memory' : 'sequence'}
                </span>
              </div>
            )}
            <div
              className="frag flex flex-col rounded-lg border border-mist bg-paper p-5 shadow-sm"
              data-visible={visible[i]}
            >
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-[11.5px] font-bold tracking-[0.13em] text-teal-ink">
                  TIER {t.n}
                </span>
              </div>
              <h3 className="font-display mt-1 text-[22px] leading-tight font-bold text-navy">{t.name}</h3>
              <p className="mt-0.5 text-[13.5px] font-semibold text-orange-ink">{t.role}</p>
              <p className="mt-3 text-[15px] leading-snug text-ink">{t.plain}</p>
              <ul className="mt-3 space-y-1.5 border-t border-mist pt-3">
                {t.details.map((d) => (
                  <li key={d} className="flex gap-2 text-[12.5px] leading-snug text-muted">
                    <span className="mt-[7px] h-[3px] w-2 shrink-0 rounded bg-mist" />
                    {d}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
      <div
        className="frag flex items-center justify-between rounded-lg bg-navy px-6 py-3 font-mono text-[13.5px] text-white"
        data-visible={showLoss}
      >
        <span>{ARCHITECTURE.loss}</span>
        <span className="text-teal">{ARCHITECTURE.specs}</span>
      </div>
    </div>
  );
}
