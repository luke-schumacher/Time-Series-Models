import type { ReactNode } from 'react';

/** Headline KPI card — big Archivo numeral, label, provenance subline. */
export function MetricCard({
  value,
  unit,
  label,
  sub,
  accent = 'teal',
}: {
  value: string;
  unit?: string;
  label: string;
  sub?: string;
  accent?: 'teal' | 'navy' | 'orange';
}) {
  const top = { teal: 'border-t-teal', navy: 'border-t-navy', orange: 'border-t-orange' }[accent];
  const size = value.length > 6 ? 'text-[43px]' : 'text-[52px]';
  return (
    <div className={`flex flex-col rounded-lg border border-mist bg-paper p-6 shadow-sm ${top} border-t-4`}>
      <div className={`kpi-number ${size} leading-none font-bold whitespace-nowrap text-navy`}>
        {value}
        {unit && <span className="ml-1 text-[24px] font-semibold text-muted">{unit}</span>}
      </div>
      <div className="mt-3 text-[17px] leading-tight font-semibold text-ink">{label}</div>
      {sub && <div className="mt-1.5 text-[14px] leading-snug text-muted">{sub}</div>}
    </div>
  );
}

/** Compact stat tile (data-foundation style). */
export function StatTile({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div className="rounded-lg bg-surface px-5 py-4">
      <div className="kpi-number text-[34px] leading-none font-bold text-teal-ink">{value}</div>
      <div className="mt-1.5 text-[15px] font-semibold text-ink">{label}</div>
      {sub && <div className="mt-0.5 font-mono text-[12px] text-muted">{sub}</div>}
    </div>
  );
}

export function StatusChip({ state }: { state: 'done' | 'refining' | 'planned' | 'ready' | 'progress' }) {
  const map = {
    done: { label: 'DONE', cls: 'bg-teal/12 text-teal-ink border-teal/40', dot: 'bg-teal' },
    ready: { label: 'READY', cls: 'bg-teal/12 text-teal-ink border-teal/40', dot: 'bg-teal' },
    refining: { label: 'REFINING', cls: 'bg-orange/10 text-orange-ink border-orange/40', dot: 'bg-orange' },
    progress: { label: 'IN PROGRESS', cls: 'bg-orange/10 text-orange-ink border-orange/40', dot: 'bg-orange' },
    planned: { label: 'PLANNED', cls: 'bg-surface text-muted border-mist', dot: 'bg-muted' },
  } as const;
  const m = map[state];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[11px] font-semibold tracking-wide ${m.cls}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${m.dot}`} />
      {m.label}
    </span>
  );
}

/** Journey phase card (Act 2). */
export function PhaseCard({
  n,
  phase,
  desc,
  outcome,
  chip,
}: {
  n: number;
  phase: string;
  desc: string;
  outcome: string;
  chip?: ReactNode;
}) {
  return (
    <div className="flex flex-col rounded-lg border border-mist bg-paper p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[12px] font-semibold tracking-[0.12em] text-teal-ink">
          PHASE {n}
        </span>
        {chip}
      </div>
      <h3 className="font-display mt-2 text-[21px] leading-tight font-bold text-navy">{phase}</h3>
      <p className="mt-2 flex-1 text-[14.5px] leading-snug text-ink/85">{desc}</p>
      <p className="mt-3 border-t border-mist pt-2.5 text-[13.5px] leading-snug font-semibold text-teal-ink">
        {outcome}
      </p>
    </div>
  );
}
