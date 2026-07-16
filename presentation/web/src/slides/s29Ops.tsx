import { StatusChip } from '../components/cards';
import { RedCallout, SlideTitle } from '../components/core';
import { OPS } from '../data/facts';

function EnvCard({ env, accent }: { env: typeof OPS.local | typeof OPS.cloud; accent: 'navy' | 'teal' }) {
  return (
    <div className={`rounded-lg border bg-paper shadow-sm ${accent === 'teal' ? 'border-teal/40' : 'border-mist'}`}>
      <div className={`rounded-t-lg px-5 py-2.5 text-[15.5px] font-semibold text-white ${accent === 'teal' ? 'bg-teal' : 'bg-navy'}`}>
        {env.title}
      </div>
      <dl className="space-y-2 p-5">
        {env.rows.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[120px_1fr] gap-3 text-[14px] leading-snug">
            <dt className="font-mono text-[12px] tracking-wide text-muted uppercase">{k}</dt>
            <dd className="text-ink/90">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function S29Ops() {
  return (
    <>
      <SlideTitle lede="One codebase, two execution worlds — already true today, not a migration plan.">
        How it runs: local ↔ Databricks parity
      </SlideTitle>
      <div className="grid grid-cols-2 gap-6">
        <EnvCard env={OPS.local} accent="navy" />
        <EnvCard env={OPS.cloud} accent="teal" />
      </div>
      <div className="mt-5 space-y-2">
        {OPS.status.map((s) => (
          <div key={s.step} className="flex items-center gap-4 rounded-md bg-surface px-4 py-2">
            <span className="w-20 font-mono text-[12.5px] font-bold text-navy">{s.step}</span>
            <span className="flex-1 text-[14.5px] text-ink/85">{s.label}</span>
            <StatusChip state={s.state as 'done' | 'refining' | 'planned'} />
          </div>
        ))}
      </div>
      <RedCallout>{OPS.parity}</RedCallout>
    </>
  );
}
