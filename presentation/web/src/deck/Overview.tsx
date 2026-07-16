import { useDeck } from './DeckContext';

/** Esc/O — act-grouped jump grid. */
export function Overview() {
  const { slides, acts, index, goTo, overview } = useDeck();
  if (!overview) return null;

  return (
    <div className="absolute inset-0 z-40 overflow-y-auto bg-navy-deep/[0.97] px-16 py-10 text-white">
      <div className="mb-6 flex items-baseline justify-between">
        <h2 className="font-display text-[26px] font-bold">Overview</h2>
        <div className="font-mono text-[12px] text-white/50">
          click to jump · Esc to close
        </div>
      </div>
      <div className="space-y-5">
        {acts.map((act) => {
          const actSlides = slides
            .map((s, i) => ({ s, i }))
            .filter(({ s }) => s.act === act.id);
          if (actSlides.length === 0) return null;
          return (
            <div key={act.id}>
              <div className="mb-2 flex items-baseline gap-3">
                <span className="font-mono text-[12px] tracking-[0.14em] text-teal uppercase">
                  {act.minutes > 0 ? `Act ${act.id} · ${act.label}` : act.label}
                </span>
                {act.minutes > 0 && (
                  <span className="font-mono text-[11px] text-white/40">{act.minutes} min</span>
                )}
              </div>
              <div className="grid grid-cols-5 gap-2">
                {actSlides.map(({ s, i }) => (
                  <button
                    key={s.id}
                    onClick={() => goTo(i)}
                    className={`rounded-md border px-3 py-2.5 text-left transition-colors ${
                      i === index
                        ? 'border-teal bg-teal/20'
                        : 'border-white/15 bg-white/[0.04] hover:border-teal/60 hover:bg-white/[0.09]'
                    }`}
                  >
                    <div className="font-mono text-[10px] text-white/45">
                      {s.appendix ? s.id.toUpperCase() : i + 1}
                    </div>
                    <div className="mt-0.5 line-clamp-2 text-[12.5px] leading-snug font-medium">
                      {s.title}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
