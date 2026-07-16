import type { ReactNode } from 'react';
import { TodoChip } from './core';

/** Deck-standard table: navy header, striped surface rows. */
export function SHSTable({
  head,
  rows,
  mono = [],
  className = '',
}: {
  head: readonly ReactNode[];
  rows: readonly (readonly ReactNode[])[];
  /** column indices rendered in the mono face */
  mono?: number[];
  className?: string;
}) {
  return (
    <table className={`w-full border-separate border-spacing-0 overflow-hidden rounded-lg text-left ${className}`}>
      <thead>
        <tr>
          {head.map((h, i) => (
            <th
              key={i}
              className="bg-navy px-4 py-2.5 text-[14px] font-semibold tracking-wide text-white first:rounded-tl-lg last:rounded-tr-lg"
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, ri) => (
          <tr key={ri} className={ri % 2 === 0 ? 'bg-surface' : 'bg-paper'}>
            {r.map((c, ci) => (
              <td
                key={ci}
                className={`border-b border-mist px-4 py-2.5 align-top text-[14.5px] leading-snug ${
                  mono.includes(ci) ? 'font-mono text-[13px]' : ''
                }`}
              >
                {c}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Two-column contrast block: the left is the status quo, the right is ours. */
export function CompareColumns({
  left,
  right,
}: {
  left: { title: string; items: readonly ReactNode[] };
  right: { title: string; items: readonly ReactNode[] };
}) {
  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="rounded-lg border border-mist bg-paper">
        <div className="rounded-t-lg bg-muted/90 px-5 py-2.5 text-[15.5px] font-semibold text-white">
          {left.title}
        </div>
        <ul className="space-y-2.5 p-5 text-[15.5px] leading-snug">
          {left.items.map((it, i) => (
            <li key={i} className="flex gap-2.5 text-ink/80">
              <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-muted/50" />
              <span>{it}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="rounded-lg border border-teal/40 bg-paper shadow-sm">
        <div className="rounded-t-lg bg-teal px-5 py-2.5 text-[15.5px] font-semibold text-white">
          {right.title}
        </div>
        <ul className="space-y-2.5 p-5 text-[15.5px] leading-snug">
          {right.items.map((it, i) => (
            <li key={i} className="flex gap-2.5">
              <span className="mt-[9px] h-1 w-3 shrink-0 rounded bg-teal" />
              <span>{it}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Labeled drop-in frame for real screenshots that don't exist yet. */
export function ScreenshotSlot({
  name,
  hint,
  children,
  className = '',
}: {
  name: string;
  hint: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-mist bg-surface/60 p-6 text-center ${className}`}
    >
      {children ?? (
        <>
          <svg width="34" height="34" viewBox="0 0 24 24" fill="none" className="text-muted/70" aria-hidden>
            <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="8.5" cy="10" r="1.6" stroke="currentColor" strokeWidth="1.5" />
            <path d="M4 17l5-4 3.5 2.8L17 11l3 3.4" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          <div className="text-[15px] font-semibold text-ink/80">{name}</div>
          <div className="max-w-[420px] text-[13px] leading-snug text-muted">{hint}</div>
          <TodoChip>drop file in public/assets/screenshots/</TodoChip>
        </>
      )}
    </div>
  );
}

export function RoadmapColumn({
  q,
  title,
  items,
  now,
}: {
  q: string;
  title: string;
  items: readonly string[];
  now: boolean;
}) {
  return (
    <div
      className={`flex flex-col rounded-lg border p-5 ${
        now ? 'border-teal bg-teal/[0.05] shadow-sm' : 'border-mist bg-paper'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-[13px] font-bold tracking-wide text-navy">{q}</span>
        {now && (
          <span className="rounded-full bg-teal px-2.5 py-0.5 font-mono text-[10.5px] font-bold tracking-wider text-white">
            NOW
          </span>
        )}
      </div>
      <h3 className="font-display mt-1.5 text-[19px] leading-tight font-bold text-navy">{title}</h3>
      <ul className="mt-3 space-y-1.5">
        {items.map((it) => (
          <li key={it} className="flex gap-2 text-[13.5px] leading-snug text-ink/85">
            <span className={`mt-[8px] h-1 w-2.5 shrink-0 rounded ${now ? 'bg-teal' : 'bg-mist'}`} />
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}
