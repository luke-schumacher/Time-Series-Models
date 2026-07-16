import { SoWhatBar, SlideTitle } from '../components/core';
import { CONTRIBUTIONS, TEAM } from '../data/facts';

export function S13Contributions() {
  return (
    <>
      <SlideTitle lede={<>Ownership across the programme · {TEAM}</>}>
        Where my work sits
      </SlideTitle>
      <div className="grid flex-1 grid-cols-3 gap-5">
        {CONTRIBUTIONS.map((c) => (
          <div
            key={c.title}
            className={`flex flex-col rounded-lg border p-5 ${
              c.mine ? 'border-t-4 border-mist border-t-teal bg-paper shadow-sm' : 'border-mist bg-surface/60'
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span
                className={`rounded-full px-2.5 py-0.5 font-mono text-[10.5px] font-bold tracking-wider ${
                  c.mine ? 'bg-teal text-white' : 'bg-muted/20 text-muted'
                }`}
              >
                {c.mine ? 'MY BUILD' : 'TEAM'}
              </span>
              {'thesis' in c && c.thesis && (
                <span className="rounded-full bg-navy px-2.5 py-0.5 font-mono text-[10.5px] font-bold tracking-wider text-white">
                  MASTER THESIS
                </span>
              )}
            </div>
            <h3 className={`font-display mt-2.5 text-[19px] leading-tight font-bold ${c.mine ? 'text-navy' : 'text-muted'}`}>
              {c.title}
            </h3>
            <p className={`mt-2 flex-1 text-[14px] leading-snug ${c.mine ? 'text-ink/85' : 'text-muted'}`}>
              {c.desc}
            </p>
            {c.title === 'MRRT Insight Agent' && (
              <p className="mt-2.5 font-mono text-[11.5px] text-teal-ink">
                MRRT corpus via Sebastian
              </p>
            )}
          </div>
        ))}
      </div>
      <SoWhatBar>
        The generative core, its cloud path, and both AI agents are my direct work — handover-ready
        and documented.
      </SoWhatBar>
    </>
  );
}
