import { RedCallout, SlideTitle } from '../components/core';
import { ACTS } from '../deck/acts';

export function S03Agenda() {
  const acts = ACTS.filter((a) => a.minutes > 0 && a.id > 0);
  return (
    <>
      <SlideTitle>How the next hour runs</SlideTitle>
      <div className="grid grid-cols-2 gap-x-12 gap-y-4">
        {acts.map((a) => (
          <div key={a.id} className="flex items-center gap-4 border-b border-mist pb-3.5">
            <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-navy font-mono text-[15px] font-bold text-white">
              {a.id}
            </span>
            <span className="flex-1 text-[19px] font-semibold text-ink">{a.label}</span>
            <span className="rounded-full bg-teal/10 px-3 py-1 font-mono text-[12.5px] font-semibold text-teal-ink">
              {a.minutes} min
            </span>
          </div>
        ))}
        <div className="flex items-center gap-4 pb-3.5">
          <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-surface font-mono text-[15px] font-bold text-muted">
            +
          </span>
          <span className="flex-1 text-[19px] font-semibold text-muted">Discussion &amp; Q&amp;A</span>
          <span className="rounded-full bg-orange/10 px-3 py-1 font-mono text-[12.5px] font-semibold text-orange-ink">
            15 min
          </span>
        </div>
      </div>
      <RedCallout label="Reading the room">
        The bar at the top of every slide is this schedule — you always know where we are and what
        is left.
      </RedCallout>
    </>
  );
}
