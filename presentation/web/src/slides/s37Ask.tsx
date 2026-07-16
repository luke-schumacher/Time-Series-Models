import { SoWhatBar, SlideTitle } from '../components/core';
import { SUMMARY } from '../data/facts';

export function S37Ask() {
  return (
    <>
      <SlideTitle lede="Four decisions unblock the next phase — none requires new budget for model work.">
        The ask
      </SlideTitle>
      <div className="grid flex-1 grid-cols-2 content-center gap-6">
        {SUMMARY.ask.map((a, i) => (
          <div key={a.title} className="flex items-center gap-5 rounded-lg border border-mist bg-paper p-6 shadow-sm">
            <span className="grid h-12 w-12 shrink-0 place-items-center rounded-full bg-orange font-mono text-[17px] font-bold text-white">
              {i + 1}
            </span>
            <div>
              <h3 className="font-display text-[22px] leading-tight font-bold text-navy">{a.title}</h3>
              <p className="mt-1 text-[15.5px] leading-snug text-ink/80">{a.desc}</p>
            </div>
          </div>
        ))}
      </div>
      <SoWhatBar>
        Decision requested today: the pilot site and the data agreement — the other two follow by
        calendar.
      </SoWhatBar>
    </>
  );
}
