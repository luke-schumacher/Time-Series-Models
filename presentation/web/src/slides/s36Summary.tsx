import { RedCallout, SlideTitle } from '../components/core';
import { SUMMARY } from '../data/facts';

export function S36Summary() {
  return (
    <>
      <SlideTitle>Three things to remember</SlideTitle>
      <div className="flex flex-1 flex-col justify-center gap-6">
        {SUMMARY.remember.map((r, i) => (
          <div key={r.title} className="flex items-start gap-6">
            <span className="kpi-number w-16 shrink-0 text-right text-[56px] leading-none font-extrabold text-teal">
              {i + 1}
            </span>
            <div className="border-l-2 border-mist pl-6">
              <h3 className="font-display text-[24px] leading-tight font-bold text-navy">{r.title}</h3>
              <p className="mt-1.5 max-w-[1050px] text-[16.5px] leading-snug text-ink/85">{r.body}</p>
            </div>
          </div>
        ))}
      </div>
      <RedCallout>
        If you keep only one: it works, it is measured, and it runs on 40 real scanners today.
      </RedCallout>
    </>
  );
}
