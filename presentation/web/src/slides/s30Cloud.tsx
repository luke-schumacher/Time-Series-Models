import { StatusChip } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { CLOUD_READINESS } from '../data/facts';

const BENEFITS = [
  'All 40 scanners — and future fleets — processed concurrently',
  'Shared object storage feeds every downstream model and dashboard',
  'Per-customer fine-tuning becomes an orchestrated batch job',
  'Direct line to the Q2 2027 SaaS packaging on the roadmap',
] as const;

export function S30Cloud() {
  return (
    <>
      <SlideTitle lede="Readiness measured against what already runs on Databricks today.">
        Cloud potential — scale-up, not rewrite
      </SlideTitle>
      <div className="grid flex-1 grid-cols-[1.35fr_1fr] gap-8">
        <div className="flex flex-col justify-center gap-2">
          {CLOUD_READINESS.map((r) => (
            <div key={r.item} className="flex items-center gap-4 rounded-md border border-mist bg-paper px-4 py-2.5">
              <StatusChip state={r.state as 'ready' | 'progress' | 'planned'} />
              <span className="flex-1 text-[15px] font-medium text-ink">{r.item}</span>
              <span className="font-mono text-[12px] text-muted">{r.note}</span>
            </div>
          ))}
        </div>
        <div className="flex flex-col justify-center">
          <div className="rounded-lg bg-navy p-6 text-white shadow-md">
            <div className="font-mono text-[11.5px] tracking-[0.13em] text-teal uppercase">
              What full cloud operation buys
            </div>
            <ul className="mt-4 space-y-3">
              {BENEFITS.map((b) => (
                <li key={b} className="flex gap-2.5 text-[15px] leading-snug text-white/90">
                  <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-teal" />
                  {b}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      <RedCallout>
        The same model weights already run in both worlds — moving to cloud changes where the
        pipeline runs, not what it is.
      </RedCallout>
    </>
  );
}
