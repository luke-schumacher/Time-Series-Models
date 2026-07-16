import { RedCallout, SlideTitle } from '../components/core';
import { PROBLEM } from '../data/facts';
import { DurationRanges } from '../viz/DurationRanges';

export function S05Problem() {
  return (
    <>
      <SlideTitle lede={`One scanner serves scans from ${PROBLEM.scanRange} — static appointment slots cannot adapt.`}>
        The scheduling problem
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.15fr_1fr] gap-10">
        <ul className="space-y-4 text-[18px] leading-snug">
          {[
            <>A single overrun <strong>cascades</strong> — delays ripple through the entire day</>,
            <>Clinical priority (urgent vs routine) further disrupts static queues</>,
            <>Technician workload spikes unpredictably at coil changes</>,
            <>Post-hoc reports describe yesterday; nothing simulates tomorrow</>,
          ].map((item, i) => (
            <li key={i} className="flex gap-3">
              <span className="mt-[11px] h-1.5 w-4 shrink-0 rounded bg-teal" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
        <div className="flex flex-col justify-center">
          <DurationRanges />
          <p className="mt-1 text-right font-mono text-[11.5px] text-muted">
            per-scan ranges on one scanner
          </p>
        </div>
      </div>
      <RedCallout label="Root cause">{PROBLEM.rootCause}</RedCallout>
    </>
  );
}
