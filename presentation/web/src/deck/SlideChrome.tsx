import type { ReactNode } from 'react';
import { useDeck } from './DeckContext';
import { talkProgress } from './logic';

/**
 * Signature element: the talk itself rendered as a scanner-day schedule.
 * Act blocks are sized by their time budget; the teal scan line is "now".
 */
function ScheduleBar() {
  const { acts, slides, index, fragment } = useDeck();
  const core = acts.filter((a) => a.minutes > 0);
  const total = core.reduce((s, a) => s + a.minutes, 0);
  const isAppendix = Boolean(slides[index]?.appendix);
  const pos = talkProgress({ index, fragment }, slides);

  return (
    <div
      className={`relative flex h-1.5 w-full gap-[3px] ${isAppendix ? 'opacity-40' : ''}`}
      aria-hidden="true"
    >
      {core.map((a) => (
        <div
          key={a.id}
          className="h-full rounded-[2px] bg-mist"
          style={{ width: `${(a.minutes / total) * 100}%` }}
          title={`${a.label} — ${a.minutes} min`}
        />
      ))}
      {/* filled portion */}
      <div
        className="absolute top-0 left-0 h-full rounded-[2px] bg-teal/70"
        style={{ width: `${pos * 100}%`, transition: 'width 300ms ease' }}
      />
      {/* scan line */}
      <div
        className="scanline absolute -top-[3px] h-[12px] w-[2.5px] rounded bg-teal"
        style={{ left: `calc(${pos * 100}% - 1px)`, transition: 'left 300ms ease' }}
      />
    </div>
  );
}

export function SlideChrome({ children }: { children: ReactNode }) {
  const { slide, slides, acts, index } = useDeck();
  const act = acts.find((a) => a.id === slide.act);
  const coreCount = slides.filter((s) => !s.appendix).length;
  const counter = slide.appendix
    ? `APPENDIX · ${slide.id.toUpperCase()}`
    : `${index + 1} / ${coreCount}`;

  if (slide.chrome === 'hero') {
    return (
      <div className="absolute inset-0">
        {children}
        <div className="absolute right-8 bottom-6 font-mono text-[13px] tracking-wide text-white/50">
          {counter}
        </div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 flex flex-col bg-paper">
      {/* identity edge */}
      <div className="absolute top-0 bottom-0 left-0 w-[6px] bg-teal" />
      <div className="flex flex-1 flex-col px-20 pt-9 pb-12">
        <ScheduleBar />
        <div className="mt-4 flex items-baseline justify-between">
          <div className="font-mono text-[13px] font-medium tracking-[0.14em] text-muted uppercase">
            {slide.appendix ? 'Appendix' : `Act ${slide.act} · ${act?.label ?? ''}`}
          </div>
          <div className="font-mono text-[13px] tracking-wide text-muted">{counter}</div>
        </div>
        <div className="slide-enter mt-5 flex min-h-0 flex-1 flex-col">{children}</div>
      </div>
    </div>
  );
}
