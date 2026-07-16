import { useEffect, useState } from 'react';
import { useDeck } from './DeckContext';

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

/** N — presenter drawer: notes, act budget, elapsed clock, next slide. */
export function SpeakerNotes() {
  const { slide, slides, acts, index, notesOpen, startedAt } = useDeck();
  const [, tick] = useState(0);

  useEffect(() => {
    if (!notesOpen || startedAt === null) return;
    const t = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(t);
  }, [notesOpen, startedAt]);

  if (!notesOpen) return null;

  const act = acts.find((a) => a.id === slide.act);
  const nextSlide = slides[index + 1];
  const budgetToHere = acts
    .filter((a) => a.minutes > 0 && a.id <= slide.act)
    .reduce((s, a) => s + a.minutes, 0);

  return (
    <div className="absolute top-0 right-0 bottom-0 z-30 flex w-[430px] flex-col bg-navy-deep/[0.96] p-7 text-white">
      <div className="flex items-baseline justify-between">
        <span className="font-mono text-[12px] tracking-[0.14em] text-teal uppercase">
          Presenter notes
        </span>
        <span className="font-mono text-[12px] text-white/50">N to close</span>
      </div>

      <h3 className="font-display mt-4 text-[20px] leading-snug font-bold">{slide.title}</h3>

      <p className="mt-4 flex-1 overflow-y-auto text-[15.5px] leading-relaxed text-white/85">
        {slide.notes}
      </p>

      <div className="mt-5 space-y-2.5 border-t border-white/15 pt-4">
        <div className="flex justify-between font-mono text-[13px]">
          <span className="text-white/55">Elapsed</span>
          <span className="text-teal">{startedAt ? formatElapsed(Date.now() - startedAt) : '—'}</span>
        </div>
        <div className="flex justify-between font-mono text-[13px]">
          <span className="text-white/55">Budget by end of act {slide.act}</span>
          <span>{act && act.minutes > 0 ? `${budgetToHere} min` : 'n/a'}</span>
        </div>
        <div className="flex justify-between gap-6 font-mono text-[13px]">
          <span className="shrink-0 text-white/55">Next</span>
          <span className="truncate text-right text-white/85">
            {nextSlide ? nextSlide.title : 'end of deck'}
          </span>
        </div>
      </div>
    </div>
  );
}
