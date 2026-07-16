import type { ReactNode } from 'react';

export function SlideTitle({ children, lede }: { children: ReactNode; lede?: ReactNode }) {
  return (
    <header className="mb-6">
      <h2 className="font-display text-[38px] leading-[1.08] font-bold text-navy">{children}</h2>
      {lede && <p className="mt-2 max-w-[1100px] text-[19px] text-muted">{lede}</p>}
    </header>
  );
}

/**
 * The "red one-liner" context callout (Telekom-style structure, SHS orange).
 * One per slide, bottom-anchored: short enough to scan, complete enough to stand alone.
 */
export function RedCallout({ label = 'The point', children }: { label?: string; children: ReactNode }) {
  return (
    <p className="mt-auto flex items-baseline gap-4 rounded-r-md border-l-[5px] border-orange bg-orange/[0.06] px-5 py-3.5">
      <span className="font-mono text-[11.5px] font-semibold tracking-[0.14em] whitespace-nowrap text-orange-ink uppercase">
        {label}
      </span>
      <span className="text-[18px] leading-snug font-medium text-ink">{children}</span>
    </p>
  );
}

/** Act-ending decision line — the Duarte "so what". */
export function SoWhatBar({ children }: { children: ReactNode }) {
  return (
    <div className="mt-auto flex items-baseline gap-4 rounded-md bg-navy px-5 py-4 text-white">
      <span className="font-mono text-[11.5px] font-semibold tracking-[0.14em] whitespace-nowrap text-teal uppercase">
        So what
      </span>
      <span className="text-[19px] leading-snug font-semibold">{children}</span>
    </div>
  );
}

/** Visible placeholder for facts the handover does not contain. */
export function TodoChip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded border border-dashed border-orange/70 bg-orange/[0.06] px-2 py-0.5 align-middle font-mono text-[11px] font-medium text-orange-ink">
      <span className="font-semibold">TODO</span>
      {children}
    </span>
  );
}

export function Pill({ children, tone = 'teal' }: { children: ReactNode; tone?: 'teal' | 'navy' | 'muted' }) {
  const tones = {
    teal: 'bg-teal/10 text-teal-ink border-teal/30',
    navy: 'bg-navy/[0.06] text-navy border-navy/25',
    muted: 'bg-surface text-muted border-mist',
  } as const;
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 font-mono text-[11.5px] font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
