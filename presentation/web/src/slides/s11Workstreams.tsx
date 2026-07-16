import { RedCallout, SlideTitle } from '../components/core';
import { WORKSTREAMS } from '../data/facts';

export function S11Workstreams() {
  return (
    <>
      <SlideTitle>Three parallel workstreams</SlideTitle>
      <div className="grid flex-1 grid-cols-3 gap-6">
        {WORKSTREAMS.map((w) => (
          <div key={w.n} className="flex flex-col rounded-lg border border-mist bg-paper p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <span className="grid h-9 w-9 place-items-center rounded-md bg-navy font-mono text-[14px] font-bold text-white">
                {w.n}
              </span>
              <h3 className="font-display text-[21px] leading-tight font-bold text-navy">{w.name}</h3>
            </div>
            <ul className="mt-4 space-y-2.5">
              {w.points.map((p) => (
                <li key={p} className="flex gap-2.5 text-[15px] leading-snug text-ink/85">
                  <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-teal" />
                  {p}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <RedCallout>
        All three draw on the same event-log foundation and share model outputs through the
        Customer Twin engine.
      </RedCallout>
    </>
  );
}
